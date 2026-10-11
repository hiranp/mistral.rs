#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! EmbeddingGemma 2 (`EmbeddingGemma2Model`), text tower only.
//!
//! A Gemma 4-style stack run bidirectionally:
//! * RMSNorm uses the weight directly (not `1 + w`); `v_norm` has no scale.
//! * Attention scaling is 1.0 (q/k norms control magnitude); RoPE is half-split.
//! * Layers alternate sliding / full attention; full layers use a different head
//!   width and a single KV head (`per_layer_config`).
//! * Per-layer inputs (PLE) come from a linear projection of the scaled token
//!   embeddings only (no per-layer token table), gating each layer's output.
//! * A `512 -> 768` projection follows the final norm. Mean pooling and L2
//!   normalisation are applied by the pipeline's sentence-transformers modules.
//!
//! The vision/audio towers in the checkpoint are not loaded.

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    attention::{AttentionMask, SdpaParams},
    device_map::DeviceMapper,
    layers::{embedding, Activation, Mlp, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::BidirectionalMasker,
    paged_attention::AttentionImplementation,
    pipeline::{
        text_models_inputs_processor::FlashParams, EmbeddingModel, IsqModel, NormalLoadingMetadata,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, attention_bias, false);
serde_default_fn!(Activation, hidden_activation, Activation::GeluPytorchTanh);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(usize, vocab_size, 262144);
serde_default_fn!(usize, head_dim, 256);
serde_default_fn!(usize, sliding_window_pattern, 6);

/// Longest sequence the RoPE tables are built for. The model card's context is
/// 8192 tokens; `max_position_embeddings` (262144) would allocate ~0.5 GB of
/// tables per RoPE flavour for no benefit.
const MAX_ROPE_POSITIONS: usize = 8192;

#[derive(Debug, Clone, Deserialize)]
pub struct LayerGeometry {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeTheta {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeTheta,
    pub full_attention: RopeTheta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingGemma2TextConfig {
    #[serde(default = "attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Keyed by layer index as a string ("05", "11", ...).
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerGeometry>,
    pub rope_parameters: RopeParameters,
    pub max_position_embeddings: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "sliding_window_pattern")]
    pub sliding_window_pattern: usize,
}

/// `config.json` of the checkpoint: the text tower is nested under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: EmbeddingGemma2TextConfig,
}

impl EmbeddingGemma2TextConfig {
    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        match self.layer_types.get(layer_idx) {
            Some(t) => t == "sliding_attention",
            None => (layer_idx + 1) % self.sliding_window_pattern != 0,
        }
    }

    /// (head_dim, num_kv_heads) for a layer.
    pub fn geometry(&self, layer_idx: usize) -> (usize, usize) {
        self.per_layer_config
            .iter()
            .find(|(k, _)| k.parse::<usize>().ok() == Some(layer_idx))
            .map(|(_, g)| (g.head_dim, g.num_key_value_heads))
            .unwrap_or((self.head_dim, self.num_key_value_heads))
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_sliding_window: bool,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &EmbeddingGemma2TextConfig,
        layer_idx: usize,
        mapper: &dyn DeviceMapper,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let (head_dim, num_kv_heads) = cfg.geometry(layer_idx);
        let bias = cfg.attention_bias;

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm)?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            bias,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            bias,
            comm,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            head_dim,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;
        // `v_norm` is RMSNorm without a learned scale.
        let v_norm = RmsNorm::from_w(
            Tensor::ones(head_dim, vb.dtype(), vb.device())?,
            cfg.rms_norm_eps,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            use_sliding_window: cfg.is_sliding(layer_idx),
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    num_kv_heads,
                    num_heads,
                    comm,
                )?,
                softcap: None,
                // q/k norms control magnitude: no 1/sqrt(d).
                softmax_scale: 1.0,
                // Window semantics live in the custom mask.
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &Tensor,
        sliding_attention_mask: &Tensor,
        seqlen_offsets: &[usize],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;
        let v = v.apply(&self.v_norm)?;

        let positions =
            crate::pipeline::text_positions_tensor(seqlen_offsets, q.dim(2)?, q.device())?;
        let (q, k) = self.rotary_emb.forward(&q, &k, &positions)?;

        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::Custom(mask.clone()),
            Some(flash_params),
            &self.sdpa_params,
        )?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;
        self.o_proj.forward(&attn_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Arc<dyn QuantMethod>,
    per_layer_projection: Arc<dyn QuantMethod>,
    post_per_layer_input_norm: RmsNorm,
    layer_scalar: Tensor,
    act: Activation,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &EmbeddingGemma2TextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            layer_idx,
            mapper,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_activation,
            comm,
        )?;
        let norm = |name: &str, size: usize| {
            RmsNorm::new(
                size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp(name), false),
            )
        };
        let ple_vb = |name: &str| mapper.set_device(layer_idx, vb.pp(name), false);
        let per_layer_input_gate = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.hidden_size_per_layer_input,
            &cfg.quantization_config,
            false,
            ple_vb("ple_block").pp("per_layer_input_gate"),
        )?;
        let per_layer_projection = ReplicatedLayer::new(
            cfg.hidden_size_per_layer_input,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            ple_vb("ple_block").pp("per_layer_projection"),
        )?;
        let post_per_layer_input_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            ple_vb("ple_block").pp("post_per_layer_input_norm"),
        )?;
        let layer_scalar = mapper
            .set_device(layer_idx, vb.clone(), false)
            .get((1,), "layer_scalar")?;

        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm: norm("input_layernorm", cfg.hidden_size)?,
            post_attention_layernorm: norm("post_attention_layernorm", cfg.hidden_size)?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm", cfg.hidden_size)?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm", cfg.hidden_size)?,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            act: cfg.hidden_activation,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        attention_mask: &Tensor,
        sliding_attention_mask: &Tensor,
        seqlen_offsets: &[usize],
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self
            .self_attn
            .forward(
                &h,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        let xs = (h + residual)?;

        let residual = &xs;
        let h = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + h)?;

        // Per-layer-input block: gate -> act -> * per-layer input -> project -> norm.
        let residual = &xs;
        let g = self
            .act
            .forward(&self.per_layer_input_gate.forward(&xs)?)?
            .broadcast_mul(per_layer_input)?;
        let g = self
            .per_layer_projection
            .forward(&g)?
            .apply(&self.post_per_layer_input_norm)?;
        let xs = (residual + g)?;

        xs.broadcast_mul(&self.layer_scalar)
    }
}

pub struct EmbeddingGemma2 {
    embed_tokens: Arc<dyn QuantMethod>,
    embed_tokens_scale: f64,
    per_layer_model_projection: Arc<dyn QuantMethod>,
    per_layer_projection_norm: RmsNorm,
    per_layer_projection_scale: f64,
    embedding_projection: Arc<dyn QuantMethod>,
    dtype: DType,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    num_layers: usize,
    ple_dim: usize,
}

impl EmbeddingGemma2 {
    pub fn new(
        cfg: &EmbeddingGemma2Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let cfg = &cfg.text_config;
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            candle_core::bail!("Expected AttentionImplementation::Eager");
        }

        // The converted checkpoint nests the text tower under `language_model`.
        let vb = ["language_model", "model.language_model"]
            .into_iter()
            .find(|p| vb.contains_tensor(&format!("{p}.embed_tokens.weight")))
            .map_or_else(|| vb.clone(), |p| p.split('.').fold(vb.clone(), |v, s| v.pp(s)));

        let mapper = normal_loading_metadata.mapper;
        let dtype = vb.dtype();
        let real_device = normal_loading_metadata.real_device.clone();

        // HF multiplies by the scale cast to the weight dtype, so bf16 rounds it.
        let embed_tokens_scale = {
            let s = Tensor::new(&[(cfg.hidden_size as f32).sqrt()], &Device::Cpu)?
                .to_dtype(dtype)?
                .to_dtype(DType::F32)?;
            f64::from(s.to_vec1::<f32>()?[0])
        };

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            &cfg.quantization_config,
        )?;

        let ple_total = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        let per_layer_model_projection = ReplicatedLayer::new(
            cfg.hidden_size,
            ple_total,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(vb.pp("ple").pp("per_layer_model_projection"), false),
        )?;
        let per_layer_projection_norm = RmsNorm::new(
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("ple").pp("per_layer_projection_norm"), false),
        )?;

        let embedding_projection = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.embedding_dim,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(vb.pp("embedding_projection"), false),
        )?;

        let rope_positions = cfg
            .max_position_embeddings
            .map_or(MAX_ROPE_POSITIONS, |m| m.min(MAX_ROPE_POSITIONS));
        // One RoPE per (device, attention flavour): flavours differ in theta and width.
        let mut ropes: HashMap<(_, bool), Arc<RotaryEmbedding>> = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&real_device);
            let sliding = cfg.is_sliding(layer_idx);
            let key = (device.location(), sliding);
            if ropes.contains_key(&key) {
                continue;
            }
            let (head_dim, _) = cfg.geometry(layer_idx);
            let theta = if sliding {
                cfg.rope_parameters.sliding_attention.rope_theta
            } else {
                cfg.rope_parameters.full_attention.rope_theta
            };
            ropes.insert(
                key,
                Arc::new(RotaryEmbedding::new(
                    theta as f32,
                    head_dim,
                    rope_positions,
                    device,
                    is_gptx,
                    dtype,
                )?),
            );
        }

        let vb_l = vb.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&real_device);
            let rotary_emb = ropes
                .get(&(device.location(), cfg.is_sliding(layer_idx)))
                .expect("No RoPE for device location!")
                .clone();
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        Ok(Self {
            embed_tokens,
            embed_tokens_scale,
            per_layer_model_projection,
            per_layer_projection_norm,
            per_layer_projection_scale: (cfg.hidden_size as f64).powf(-0.5),
            embedding_projection,
            dtype,
            layers,
            norm,
            device: real_device,
            // Bidirectional window: attend while |q - k| <= sliding_window (HF passes
            // sliding_window + 1 to its exclusive-window kernels).
            sliding_window: cfg.sliding_window + 1,
            num_layers: cfg.num_hidden_layers,
            ple_dim: cfg.hidden_size_per_layer_input,
            mapper,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)? * self.embed_tokens_scale
    }

    /// `[b, t, layers, ple_dim]` per-layer inputs from the scaled embeddings.
    fn per_layer_inputs(&self, embeds: &Tensor) -> Result<Tensor> {
        let (b, t, _) = embeds.dims3()?;
        let proj = (self.per_layer_model_projection.forward(embeds)?
            * self.per_layer_projection_scale)?;
        proj.reshape((b, t, self.num_layers, self.ple_dim))?
            .apply(&self.per_layer_projection_norm)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        mut xs: Tensor,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (bs, _seqlen) = input_ids.dims2()?;
        let seqlen_offsets = vec![0; bs];

        let attention_mask = BidirectionalMasker.make_mask(input_ids, xs.dtype())?;
        let sliding_mask =
            BidirectionalMasker.make_sliding_mask(input_ids, xs.dtype(), self.sliding_window)?;
        let per_layer_inputs = self.per_layer_inputs(&xs)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let ple = per_layer_inputs
                .narrow(2, i, 1)?
                .squeeze(2)?
                .to_device(xs.device())?;
            xs = layer.forward(
                &xs,
                &ple,
                &attention_mask.to_device(xs.device())?,
                &sliding_mask.to_device(xs.device())?,
                &seqlen_offsets,
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?.apply(&self.norm)?;
        // Linear, so projecting per token equals projecting after mean pooling.
        self.embedding_projection.forward(&xs)
    }
}

impl IsqModel for EmbeddingGemma2 {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("embed_tokens").add(&self.embed_tokens);
        uvb.pp("norm").add(&self.norm);
        uvb.pp("embedding_projection")
            .add(&self.embedding_projection);
        uvb.pp("ple")
            .pp("per_layer_model_projection")
            .add(&self.per_layer_model_projection);
        uvb.pp("ple")
            .pp("per_layer_projection_norm")
            .add(&self.per_layer_projection_norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(layer_idx);
            let attn = uvb_l.pp("self_attn");
            attn.pp("q_norm").add(&layer.self_attn.q_norm);
            attn.pp("k_norm").add(&layer.self_attn.k_norm);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm);
            uvb_l
                .pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm);
            let ple = uvb_l.pp("ple_block");
            ple.pp("per_layer_input_gate")
                .add(&layer.per_layer_input_gate);
            ple.pp("per_layer_projection")
                .add(&layer.per_layer_projection);
            ple.pp("post_per_layer_input_norm")
                .add(&layer.post_per_layer_input_norm);
            uvb_l.add_tensor("layer_scalar", layer.layer_scalar.clone());
        }

        uvb.to_safetensors()
    }
}

impl EmbeddingModel for EmbeddingGemma2 {
    fn forward(
        &self,
        input_ids: &Tensor,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        self.forward_embeds(input_ids, self.embed_tokens(input_ids)?, flash_params)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for EmbeddingGemma2 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_json() -> &'static str {
        r#"{
          "architectures": ["EmbeddingGemma2Model"],
          "model_type": "embedding_gemma2",
          "text_config": {
            "attention_bias": false, "embedding_dim": 768, "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 512,
            "hidden_size_per_layer_input": 512, "intermediate_size": 2048,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention",
              "sliding_attention","sliding_attention","full_attention",
              "sliding_attention","sliding_attention","sliding_attention",
              "sliding_attention","sliding_attention","full_attention",
              "sliding_attention","sliding_attention","sliding_attention",
              "sliding_attention","sliding_attention","full_attention",
              "sliding_attention","sliding_attention","sliding_attention",
              "sliding_attention","sliding_attention","full_attention"],
            "max_position_embeddings": 262144, "num_attention_heads": 4,
            "num_hidden_layers": 24, "num_key_value_heads": 2,
            "per_layer_config": {
              "05": {"head_dim": 512, "num_key_value_heads": 1},
              "11": {"head_dim": 512, "num_key_value_heads": 1},
              "17": {"head_dim": 512, "num_key_value_heads": 1},
              "23": {"head_dim": 512, "num_key_value_heads": 1}},
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
              "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
              "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
            "sliding_window": 512, "vocab_size": 262144
          }
        }"#
    }

    #[test]
    fn parses_checkpoint_config_and_per_layer_geometry() {
        let cfg: EmbeddingGemma2Config = serde_json::from_str(cfg_json()).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 24);
        assert!(t.is_sliding(0) && t.is_sliding(4) && !t.is_sliding(5));
        assert!(!t.is_sliding(23));
        assert_eq!(t.geometry(0), (256, 2));
        assert_eq!(t.geometry(5), (512, 1));
        assert_eq!(t.geometry(23), (512, 1));
    }
}
