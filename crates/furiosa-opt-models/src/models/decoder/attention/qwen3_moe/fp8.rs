//! Qwen3-30B-A3B with block-scaled E4M3 weights on one chip: the checkpoint contract, the served
//! geometry and the registry line of the model a launch runs.
//!
//! - The experts keep their 128 x 128 block-scaled E4M3 codes in HBM, 29 of the model's 30.5
//!   billion parameters; each projects a row with its activations quantized at run time per group
//!   of 128 inputs, as the checkpoint's dynamic activation scheme states.
//! - The attention projections are dequantized to BF16 at load, a block's codes times its scale,
//!   and run as the BF16 Qwen3 layers' do: under 1 GB of weights, on the dense kernels every
//!   Qwen3 width already runs.
//! - The checkpoint's four key and value heads are each read twice, as eight heads of four query
//!   heads each: a query head attends the same keys and values as in the checkpoint, on the
//!   attention geometry of Qwen3's other sizes, at twice the cache.
//! - The router's gate, the norms, the embedding and the head are stored unquantized.

use super::{ExpertCount, LAYERS, Selected};
use crate::load::{Checkpoint, Context, Load, LoadError, Named};
use crate::models::config::{self, Quantization, Scheme, scaling::Spectrum};
use crate::models::decoder::attention::qwen3::{self, launch};
use crate::models::decoder::attention::{
    Geometry, Head, KvHeads, PaddedVocabulary, Qwen2ForCausalLM, Qwen2Layer, Windows,
};
use crate::models::decoder::{Form, form::Affine};
use crate::models::{Class, Family, Implements};
use crate::modules::feedforward::moe::Sparse;
use crate::modules::generic::collection::Array;
use crate::ops::attention::geometry::{self, ProjectionWidth, QueryGroups, QueryWidth};
use crate::ops::index::{gather::Gather, scatter::Scatter};
use crate::ops::pointwise::add::Add;
use crate::ops::rotary::Rotary;
use crate::ops::swiglu;
use crate::specialize::attention::Decode;
use crate::tensor::axis::{self, Limit};
use crate::tensor::memory::Hbm;
use furiosa_opt_std::prelude::*;
use safetensors::Dtype;
use serde::Deserialize;
use std::io;

/// HF `Qwen3MoeForCausalLM` of the block-scaled E4M3 Qwen3-30B-A3B checkpoint: the model its
/// launches run, loaded from the checkpoint with its attention projections in BF16.
#[derive(DeviceSend)]
pub struct ForCausalLM {
    pub model: Model,
}

/// The causal model a launch runs: the body of [`Layer`]s with the rotary table and greedy head
/// of Qwen3's launches.
pub type Model = launch::Over<Size, Body, Affine>;

/// The embedding, the layers and the final norm.
pub type Body = Bound::Body;

type Bound = Qwen2ForCausalLM<
    qwen3::Embed<Size, Hbm>,
    Attention,
    Position,
    Scatter,
    Gather,
    qwen3::Attend<Size>,
    Output,
    Block,
    Norm,
    Add,
    qwen3::LmHead<Size, Hbm>,
    LAYERS,
    Array,
>;

/// HF `Qwen3MoeDecoderLayer`: the BF16 attention of Qwen3's layers, then the sparse block.
pub type Layer =
    Qwen2Layer<Attention, Position, Scatter, Gather, qwen3::Attend<Size>, Output, Block, Norm, Add>;

/// A layer's query, key and value projections in BF16 and its query and key head norms.
pub type Attention = qwen3::Heads<Size, Affine, Hbm>;

type Position = Rotary<Head, Head, bf16>;

/// A layer's output projection in BF16.
pub type Output = Named<"o_proj", <Affine as Form>::Output<Queries, Hidden, Hbm>>;

/// HF `Qwen3MoeSparseMoeBlock`: the router's gate, its eight of 128 experts by renormalized
/// probability, and the experts in block-scaled E4M3.
pub type Block = Sparse<Hidden, Intermediate, ExpertCount, Selected>;

type Norm = qwen3::QwenNorm<Hidden, Hbm>;

pub type Hidden = geometry::Hidden<2048>;
pub type Intermediate = swiglu::Intermediate<768>;
type Queries = QueryWidth<4096>;

/// Qwen3-30B-A3B as the line computes it: the checkpoint's geometry with each of its key and value
/// heads read twice.
pub struct Size;

/// The key and value heads the checkpoint stores.
pub const STORED_HEADS: usize = 4;

impl Geometry for Size {
    const SIZE: &'static str = "30B-A3B";
    const FAMILY: &'static str = "qwen3_moe";
    type Queries = Queries;
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<4>;
    type Head = Head;
    type Layers<L> = [L; LAYERS];
    type Norm<W: AxisName, X: crate::tensor::memory::Memory> = qwen3::QwenNorm<W, X>;

    const LAYERS: usize = LAYERS;
    const EPSILON: usize = qwen3::EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

impl qwen3::Front for Size {
    type Heads<Projection, Norm> = crate::modules::attention::NormalizedQkv<
        Projection,
        Named<"q_norm", Norm>,
        Named<"k_norm", Norm>,
    >;
    type HeadNorm<Norm> = Norm;
}

// The rotary tiles of a launch's 1024 rows.
axes![Slots = 8];

/// A call as Qwen3-4B's, whose attention geometry this is.
impl Decode for Size {
    type Rows = Limit<16, 64>;
    type Windows = Windows<256>;
    type Width = ProjectionWidth<6144>;
    type Packed = m![{ Symbol<ProjectionWidth<6144>> }, { Symbol<Hidden> }];
    type Work = qwen3::Work<QueryGroups<4>, { qwen3::tiles::<Self>() }>;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = Slots;
    type Positions = axis::Rows<32768>;
}

const _: () = assert!(KvHeads::SIZE == 2 * STORED_HEADS);

/// The checkpoint's projections at their root, its attention ones in BF16 with each key and value
/// head twice; every name is read.
impl<'a> Load<&'a Checkpoint> for ForCausalLM {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let served = served(from)?;
        let model = Model::load(context, &served)?;
        Ok(Self { model })
    }
}

/// The attention projections of a layer, which the line runs in BF16.
const ATTENTION: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// Whether `name` is a tensor of a layer's attention projection.
fn attention(name: &str) -> bool {
    let projection = |name: &str| {
        let module = name.rsplit_once('.').map_or(name, |(module, _)| module);
        ATTENTION
            .iter()
            .any(|p| module.ends_with(&format!(".self_attn.{p}")))
    };
    projection(name)
}

/// `from` with each attention projection dequantized to BF16, its key and value heads repeated.
fn served(from: &Checkpoint) -> Result<Checkpoint, LoadError> {
    let mut served = from.without(attention);
    for layer in 0..LAYERS {
        for projection in ATTENTION {
            let at = format!("model.layers.{layer}.self_attn.{projection}");
            let (rows, columns, values) = dequantized(from, &at)?;
            let (rows, values) = match projection {
                "k_proj" | "v_proj" => (2 * rows, repeated(&values, rows / STORED_HEADS, columns)),
                _ => (rows, values),
            };
            let bytes = values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            served.insert(
                &format!("{at}.weight"),
                Dtype::BF16,
                &[rows, columns],
                bytes,
            )?;
        }
    }
    Ok(served)
}

/// The `[rows, columns]` BF16 values of the block-scaled projection `at`: each E4M3 code times its
/// 128 x 128 block's scale, rounded once.
fn dequantized(from: &Checkpoint, at: &str) -> Result<(usize, usize, Vec<bf16>), LoadError> {
    let (weight, scales) = (format!("{at}.weight"), format!("{at}.weight_scale_inv"));
    let found = |name: &str| {
        from.get(name)
            .ok_or_else(|| LoadError::missing([name.into()]))
    };
    let (codes, grid) = (found(&weight)?, found(&scales)?);
    let invalid = |name: &str, reason: String| LoadError::InvalidData {
        name: name.into(),
        reason,
    };
    let &[rows, columns] = codes.shape() else {
        return Err(invalid(
            &weight,
            format!("shape {:?}, not a matrix", codes.shape()),
        ));
    };
    if codes.dtype() != Dtype::F8_E4M3 {
        let stored = codes.dtype();
        return Err(invalid(
            &weight,
            format!("stored as {stored:?}, not E4M3 codes"),
        ));
    }
    let blocks = [rows.div_ceil(128), columns.div_ceil(128)];
    if grid.shape() != blocks {
        let shape = grid.shape();
        return Err(invalid(
            &scales,
            format!("shape {shape:?}, the blocks are {blocks:?}"),
        ));
    }
    let grid = Checkpoint::floats(&grid)
        .ok_or_else(|| invalid(&scales, format!("stored as {:?}", grid.dtype())))?;
    let values = (codes.data().iter().enumerate())
        .map(|(at, &code)| {
            let scale = grid[at / columns / 128 * blocks[1] + at % columns / 128];
            bf16::from_f32(e4m3(code) * scale)
        })
        .collect();
    Ok((rows, columns, values))
}

/// The value of the E4M3 code `bits`: a sign, four exponent bits of bias 7 and three mantissa
/// bits; no infinity, and NaN at every bit of exponent and mantissa set.
fn e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let (exponent, mantissa) = (i32::from(bits >> 3 & 0xf), f32::from(bits & 0x7));
    match (exponent, bits & 0x7f) {
        (_, 0x7f) => f32::NAN,
        (0, _) => sign * mantissa / 8.0 * 2f32.powi(-6),
        _ => sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7),
    }
}

/// Each `head` rows of `values` twice, in order.
fn repeated(values: &[bf16], head: usize, columns: usize) -> Vec<bf16> {
    (values.chunks_exact(head * columns))
        .flat_map(|rows| rows.iter().chain(rows).copied())
        .collect()
}

/// What the line reads of a checkpoint's `quantization_config`: block-scaled E4M3 weights.
#[derive(Deserialize)]
struct Declared {
    quantization_config: Option<Quantization>,
}

/// The block-scaled checkpoint of the size: its weights quantized in 128 x 128 blocks, with every
/// attention and expert projection quantized; others may stay unquantized.
pub(crate) fn contract(document: &serde_json::Value) -> io::Result<()> {
    let unsupported = |reason: String| io::Error::new(io::ErrorKind::Unsupported, reason);
    let declared: Declared = config::parse(document)?;
    let Some(Quantization {
        scheme: Scheme::Block,
        excluded,
        kv_cache: false,
    }) = declared.quantization_config
    else {
        return Err(unsupported(
            "quantization_config: Qwen3 MoE serves only block-scaled E4M3 FP8 weights".into(),
        ));
    };
    let projections = (0..LAYERS).flat_map(|layer| {
        let attention = ATTENTION.map(|p| format!("model.layers.{layer}.self_attn.{p}"));
        let experts = (0..ExpertCount::<128>::SIZE).flat_map(move |expert| {
            ["gate_proj", "up_proj", "down_proj"]
                .map(|p| format!("model.layers.{layer}.mlp.experts.{expert}.{p}"))
        });
        attention.into_iter().chain(experts)
    });
    match projections
        .into_iter()
        .find(|module| excluded.excludes(module))
    {
        Some(module) => Err(unsupported(format!(
            "quantization_config: {module} is left unquantized, the line reads its E4M3 codes"
        ))),
        None => {
            let spectrum: Spectrum = config::parse(document)?;
            spectrum.bind::<Head, Head>().map(drop)
        }
    }
}

/// Qwen3-30B-A3B reads a checkpoint at its root, and refuses one whose config is not of its size
/// or not block-scaled E4M3.
impl Implements for ForCausalLM {
    const SIZE: &'static str = Size::SIZE;
    const CLASS: Class = Class {
        family: Size::FAMILY,
        name: "ForCausalLM",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        let refused = |error: io::Error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        };
        <super::Size<"30B-A3B"> as Family>::parse(source.config()).map_err(refused)?;
        contract(source.config()).map_err(refused)?;
        Ok("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// `Qwen/Qwen3-30B-A3B-FP8`'s config.json, its exclusions those `furiosa-ai`'s copy declares
    /// for layer 0 and the head.
    pub(crate) fn published() -> Value {
        serde_json::from_str(
            r#"{
            "architectures": ["Qwen3MoeForCausalLM"], "attention_bias": false,
            "attention_dropout": 0.0, "bos_token_id": 151643, "decoder_sparse_step": 1,
            "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
            "initializer_range": 0.02, "intermediate_size": 6144, "max_position_embeddings": 40960,
            "max_window_layers": 48, "mlp_only_layers": [], "model_type": "qwen3_moe",
            "moe_intermediate_size": 768, "norm_topk_prob": true, "num_attention_heads": 32,
            "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48,
            "num_key_value_heads": 4, "output_router_logits": false, "rms_norm_eps": 1e-06,
            "rope_scaling": null, "rope_theta": 1000000.0, "router_aux_loss_coef": 0.001,
            "sliding_window": null, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
            "vocab_size": 151936,
            "quantization_config": {
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "lm_head", "model.layers.0.input_layernorm", "model.layers.0.mlp.gate",
                    "model.layers.0.post_attention_layernorm"
                ]
            }
        }"#,
        )
        .unwrap()
    }

    /// The published config is of the line; another quantization, an excluded projection or a
    /// rotation the size does not hold is refused as unsupported by its field.
    #[test]
    fn refuses_what_the_line_does_not_serve() {
        contract(&published()).unwrap();
        let refusal = |edit: &dyn Fn(&mut Value)| {
            let mut document = published();
            edit(&mut document);
            let error = contract(&document).unwrap_err();
            (error.kind(), error.to_string())
        };
        let quantized = "quantization_config: Qwen3 MoE serves only block-scaled E4M3 FP8 weights";
        let unsupported = io::ErrorKind::Unsupported;
        for (edit, reason) in [
            (
                Box::new(|d: &mut Value| d["quantization_config"] = Value::Null)
                    as Box<dyn Fn(&mut Value)>,
                quantized.to_string(),
            ),
            (
                Box::new(|d: &mut Value| {
                    d["quantization_config"] =
                        json!({"quant_method": "fp8", "activation_scheme": "static"})
                }),
                quantized.into(),
            ),
            (
                Box::new(|d: &mut Value| {
                    d["quantization_config"]["modules_to_not_convert"] =
                        json!(["model.layers.0.mlp.experts"])
                }),
                "quantization_config: model.layers.0.mlp.experts.0.gate_proj is left unquantized, \
                 the line reads its E4M3 codes"
                    .into(),
            ),
            (
                Box::new(|d: &mut Value| {
                    d["quantization_config"]["modules_to_not_convert"] =
                        json!(["model.layers.3.self_attn"])
                }),
                "quantization_config: model.layers.3.self_attn.q_proj is left unquantized, \
                 the line reads its E4M3 codes"
                    .into(),
            ),
        ] {
            assert_eq!(refusal(&*edit), (unsupported, reason.clone()), "{reason}");
        }
        let rotated = refusal(&|d| d["partial_rotary_factor"] = json!(0.5));
        assert_eq!(rotated.0, unsupported);
    }

    /// Every E4M3 code but NaN decodes to the value whose nearest code it is.
    #[test]
    fn decodes_every_code() {
        for bits in 0..=u8::MAX {
            let decoded = e4m3(bits);
            if bits & 0x7f == 0x7f {
                assert!(decoded.is_nan(), "{bits:#x}");
                continue;
            }
            let code = f8e4m3::from_f32(decoded);
            assert_eq!(f8e4m3::to_buf(&[code])[0], bits, "{bits:#x}: {decoded}");
            assert_eq!(code.to_f32().to_bits(), decoded.to_bits(), "{bits:#x}");
        }
    }

    /// A projection's E4M3 codes times its block's scale, the key and value heads each twice.
    #[test]
    fn dequantizes_attention_and_repeats_its_heads() {
        let directory = tempfile::tempdir().unwrap();
        let mut source = Checkpoint::new(directory.path(), Some(published())).unwrap();
        let (rows, columns) = (256, 256);
        let codes: Vec<u8> = (0..rows * columns)
            .map(|at| f8e4m3::to_buf(&[f8e4m3::from_f32(((at % 17) as f32 - 8.0) / 4.0)])[0])
            .collect();
        let grid: Vec<u8> = [0.5f32, 2.0, 0.25, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        source
            .insert("w.weight", Dtype::F8_E4M3, &[rows, columns], codes.clone())
            .unwrap();
        source
            .insert("w.weight_scale_inv", Dtype::F32, &[2, 2], grid)
            .unwrap();
        let (r, c, values) = dequantized(&source, "w").unwrap();
        assert_eq!((r, c), (rows, columns));
        for (row, column, scale) in [(0, 0, 0.5), (5, 200, 2.0), (130, 3, 0.25), (255, 255, 4.0)] {
            let code = e4m3(codes[row * columns + column]);
            assert_eq!(values[row * columns + column].to_f32(), code * scale);
        }
        let heads = repeated(&values, 64, columns);
        assert_eq!(heads.len(), 2 * values.len());
        let head =
            |values: &[bf16], at: usize| values[at * 64 * columns..][..64 * columns].to_vec();
        for stored in 0..4 {
            assert_eq!(head(&heads, 2 * stored), head(&values, stored));
            assert_eq!(head(&heads, 2 * stored + 1), head(&values, stored));
        }
        assert!(attention(
            "model.layers.7.self_attn.k_proj.weight_scale_inv"
        ));
        assert!(!attention("model.layers.7.self_attn.k_norm.weight"));
        assert!(!attention("model.layers.7.mlp.experts.3.up_proj.weight"));
    }
}
