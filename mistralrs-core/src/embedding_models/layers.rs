use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::Module;
use serde::Deserialize;

/// Pooling layer
///
/// Accepts both sentence-transformers config styles: the boolean
/// `pooling_mode_*_tokens` flags with `word_embedding_dimension`, and the newer
/// `pooling_mode: "mean"` string with `embedding_dimension`.
#[derive(Deserialize, Debug, Clone)]
#[serde(from = "RawPooling")]
pub struct Pooling {
    pub word_embedding_dimension: usize,
    pub pooling_mode_cls_token: bool,
    pub pooling_mode_mean_tokens: bool,
    pub pooling_mode_max_tokens: bool,
    pub pooling_mode_mean_sqrt_len_tokens: bool,
    pub pooling_mode_weightedmean_tokens: bool,
    pub pooling_mode_lasttoken: bool,
    pub include_prompt: bool,
}

#[derive(Deserialize)]
struct RawPooling {
    #[serde(alias = "embedding_dimension")]
    word_embedding_dimension: usize,
    /// Newer style: one of `cls`, `mean`, `max`, `mean_sqrt_len_tokens`,
    /// `weightedmean`, `lasttoken`.
    pooling_mode: Option<String>,
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default = "include_prompt_default")]
    include_prompt: bool,
}

fn include_prompt_default() -> bool {
    true
}

impl From<RawPooling> for Pooling {
    fn from(r: RawPooling) -> Self {
        let mode = r.pooling_mode.as_deref();
        let is = |m: &str| mode == Some(m);
        Self {
            word_embedding_dimension: r.word_embedding_dimension,
            pooling_mode_cls_token: r.pooling_mode_cls_token || is("cls"),
            pooling_mode_mean_tokens: r.pooling_mode_mean_tokens || is("mean"),
            pooling_mode_max_tokens: r.pooling_mode_max_tokens || is("max"),
            pooling_mode_mean_sqrt_len_tokens: r.pooling_mode_mean_sqrt_len_tokens
                || is("mean_sqrt_len_tokens"),
            pooling_mode_weightedmean_tokens: r.pooling_mode_weightedmean_tokens
                || is("weightedmean"),
            pooling_mode_lasttoken: r.pooling_mode_lasttoken || is("lasttoken"),
            include_prompt: r.include_prompt,
        }
    }
}

impl Module for Pooling {
    // https://github.com/huggingface/sentence-transformers/blob/85ec64559f4414aa536eca4bf53538291e0a333f/sentence_transformers/models/Pooling.py#L26
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if !self.include_prompt {
            candle_core::bail!("Only support include_prompt==true");
        }
        if xs.dim(D::Minus1)? != self.word_embedding_dimension {
            candle_core::bail!("xs does not match the expected embedding dimension.");
        }

        let mut outputs = Vec::new();
        if self.pooling_mode_cls_token {
            unimplemented!();
        }
        if self.pooling_mode_mean_tokens {
            // Assume full attention mask. Otherwise this must be updated

            let sum_embeddings = xs.sum(1)?;
            outputs.push(sum_embeddings);
        }
        if self.pooling_mode_max_tokens {
            unimplemented!();
        }
        if self.pooling_mode_mean_sqrt_len_tokens {
            unimplemented!();
        }
        if self.pooling_mode_weightedmean_tokens {
            unimplemented!();
        }
        if self.pooling_mode_lasttoken {
            outputs.push(xs.i((.., xs.dim(D::Minus2)? - 1, ..))?);
        }

        Tensor::cat(&outputs, 1)
    }
}

/// Normalize layer
#[derive(Deserialize, Debug, Clone)]
pub struct Normalize;

impl Module for Normalize {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = (xs.sqr()?.sum(1)? + 1e-12)?.sqrt()?;

        xs.broadcast_div(&norm.unsqueeze(D::Minus1)?)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub enum DenseActivation {
    #[serde(alias = "torch.nn.modules.linear.Identity")]
    Identity,
}

/// Dense layer
#[derive(Deserialize, Debug, Clone)]
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub activation_function: DenseActivation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_accepts_legacy_flags() {
        let p: Pooling = serde_json::from_str(
            r#"{"word_embedding_dimension":768,"pooling_mode_cls_token":false,
            "pooling_mode_mean_tokens":true,"pooling_mode_max_tokens":false,
            "pooling_mode_mean_sqrt_len_tokens":false,"pooling_mode_weightedmean_tokens":false,
            "pooling_mode_lasttoken":false,"include_prompt":true}"#,
        )
        .unwrap();
        assert!(p.pooling_mode_mean_tokens && !p.pooling_mode_lasttoken);
        assert_eq!(p.word_embedding_dimension, 768);
    }

    #[test]
    fn pooling_accepts_sentence_transformers_6_style() {
        // google/embeddinggemma-2/1_Pooling/config.json
        let p: Pooling = serde_json::from_str(
            r#"{"embedding_dimension":768,"pooling_mode":"mean","include_prompt":true}"#,
        )
        .unwrap();
        assert!(p.pooling_mode_mean_tokens);
        assert!(!p.pooling_mode_cls_token && !p.pooling_mode_lasttoken);
        assert_eq!(p.word_embedding_dimension, 768);
        assert!(p.include_prompt);
    }
}
