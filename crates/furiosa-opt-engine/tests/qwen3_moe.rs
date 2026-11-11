//! Qwen3 MoE reads its published checkpoints: each size parses its own config and refuses another
//! size's and what it does not serve; the block-scaled E4M3 Qwen3-30B-A3B checkpoint is the one
//! the registry serves, and a BF16 checkpoint is refused by its quantization.
//!
//! The device test loads the served checkpoint strictly and continues a prompt with the greedy
//! tokens of an independent PyTorch reference of the same checkpoint
//! (`scripts/qwen3_moe_reference.py`), whose every token leads the next by at least 4.8 logits.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::io;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, Instance, Keep, Request, Rows, Served};
use furiosa_opt_engine::registry::Implementation;
use furiosa_opt_engine::{Checkpoint, Tiling};
use furiosa_opt_models::load::LoadError;
use furiosa_opt_models::models::Family;
use furiosa_opt_models::models::decoder::attention::qwen3_moe::{Size, fp8::ForCausalLM};
use furiosa_opt_std::prelude::{Device, HostBufferSlice, Topology};
use serde_json::json;

use common::Published;

const QWEN3_30B_A3B: Published = Published::new(
    "Qwen/Qwen3-30B-A3B",
    "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
);
const QWEN3_235B_A22B: Published = Published::new(
    "Qwen/Qwen3-235B-A22B",
    "8efa61729e24bd65b1d152b5ab5409052aa80e65",
);
/// The block-scaled E4M3 checkpoint the registry serves.
const SERVED: (&str, &str) = (
    "furiosa-ai/Qwen3-30B-A3B-FP8",
    "979f37342a14e0eb4d5d943c85167f3dee065b85",
);
const QWEN3_30B_A3B_FP8: Published = Published::new(SERVED.0, SERVED.1);

/// "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10," in Qwen's tokens.
const PROMPT: [i32; 33] = [
    2507, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 220, 24, 11, 220, 16, 15, 11,
];
/// " 11, 12,": the reference's eight greedy tokens.
const CONTINUATION: [i32; 8] = [220, 16, 16, 11, 220, 16, 17, 11];

/// The size of the implementation the registry chooses under `architecture` for `source`.
fn load(architecture: &str, source: &Checkpoint) -> Result<&'static str, LoadError> {
    let (name, implementations) = Implementation::listed(&[architecture])?;
    Implementation::choose(name, implementations, source).map(Implementation::size)
}

// # Choices of the published checkpoints

/// The block-scaled E4M3 Qwen3-30B-A3B checkpoint selects its size.
#[test]
fn block_scaled_qwen3_30b_a3b_selects_its_size() {
    let checkpoint = QWEN3_30B_A3B_FP8.declared();
    assert_eq!(load("Qwen3MoeForCausalLM", &checkpoint).unwrap(), "30B-A3B");
}

/// The BF16 Qwen3-30B-A3B checkpoint is refused by its quantization: its 61 GB do not fit a chip.
#[test]
fn bf16_qwen3_30b_a3b_is_refused() {
    let checkpoint = QWEN3_30B_A3B.declared();
    match load("Qwen3MoeForCausalLM", &checkpoint) {
        Err(LoadError::Unsupported { name, reason }) => assert_eq!(
            (name.as_str(), reason.as_str()),
            (
                "config.json",
                "quantization_config: Qwen3 MoE serves only block-scaled E4M3 FP8 weights"
            )
        ),
        other => panic!("{other:?} is not unsupported"),
    }
}

// # The served checkpoint on a chip

fn served() -> Checkpoint {
    let output = Command::new("uvx")
        .args(["--from", "huggingface-hub", "hf", "download", SERVED.0])
        .args(["--revision", SERVED.1, "--quiet"])
        .args(["--include", "*.json", "--include", "*.safetensors"])
        .stderr(Stdio::inherit())
        .output()
        .expect("hf downloads the pinned checkpoint");
    assert!(output.status.success());
    let path = String::from_utf8(output.stdout).unwrap();
    Checkpoint::new(path.trim(), None).unwrap()
}

async fn request(device: &mut Device, ids: &[i32], cached: usize, blocks: &[u32]) -> Request {
    let bytes: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let ids = HostBufferSlice::from(&bytes[..])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
    Request {
        rows: vec![blocks.to_vec()],
        cached,
        tokens: Rows::new(&ids, 0..bytes.len() / 4).unwrap(),
        keep: Keep::Last,
        adapter: None,
        positions: None,
        mask: None,
        overrides: Vec::new(),
    }
}

async fn token(device: &mut Device, buffer: furiosa_opt_engine::decoder::HbmTensor) -> i32 {
    let mut bytes = vec![0; buffer.buffer.size()];
    buffer
        .buffer
        .to_host(&mut device.pdma)
        .output(&mut bytes[..])
        .await
        .unwrap();
    i32::from_le_bytes(bytes[..4].try_into().unwrap())
}

/// The served checkpoint loads strictly and continues the prompt with the reference's tokens,
/// its prompt prefilled a token a call and in one call.
#[test]
#[ignore = "opens a device and loads the pinned checkpoint"]
fn continues_the_prompt_with_the_reference_tokens() {
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = Instance::new(
        served().load::<ForCausalLM>(&mut device).unwrap(),
        Tiling::Adaptive,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for chunk in [PROMPT.len(), 1] {
            let count = (PROMPT.len() + CONTINUATION.len()).div_ceil(16);
            let blocks: Vec<u32> = (0..count as u32).rev().collect();
            let cache = Cache::new(&device, &model.limits().cache, count).unwrap();
            let (mut cached, mut selected) = (0, 0);
            for ids in PROMPT.chunks(chunk) {
                let request = request(&mut device, ids, cached, &blocks).await;
                let output = model
                    .forward(&mut device, &cache, &[request], &[])
                    .await
                    .unwrap();
                selected = token(&mut device, output).await;
                cached += ids.len();
            }
            let mut got = vec![selected];
            for _ in 1..CONTINUATION.len() {
                let request = request(&mut device, &[selected], cached, &blocks).await;
                let output = model
                    .forward(&mut device, &cache, &[request], &[])
                    .await
                    .unwrap();
                selected = token(&mut device, output).await;
                got.push(selected);
                cached += 1;
            }
            assert_eq!(got, CONTINUATION, "prefill chunk {chunk}");
        }
    });
}

// # Configs

/// Each size parses its published checkpoint's config and refuses the other size's by its width.
#[test]
fn sizes_read_their_published_configs() {
    let large = QWEN3_235B_A22B.config();
    let small = QWEN3_30B_A3B.config();
    Size::<"30B-A3B">::parse(&small).unwrap();
    Size::<"235B-A22B">::parse(&large).unwrap();
    for (error, reason) in [
        (
            Size::<"30B-A3B">::parse(&large).map(drop).unwrap_err(),
            "hidden_size: 4096, the size 30B-A3B holds 2048",
        ),
        (
            Size::<"235B-A22B">::parse(&small).map(drop).unwrap_err(),
            "hidden_size: 2048, the size 235B-A22B holds 4096",
        ),
    ] {
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (io::ErrorKind::InvalidData, reason.into()));
    }
}

/// A config the size does not serve is refused, each case by its kind and reason.
#[test]
fn size_refuses_what_it_does_not_serve() {
    use io::ErrorKind::{InvalidData, Unsupported};
    let published = QWEN3_30B_A3B.config();
    let cases = [
        (
            "moe_intermediate_size",
            json!(1536),
            InvalidData,
            "moe_intermediate_size: 1536, the size 30B-A3B holds 768",
        ),
        (
            "num_hidden_layers",
            json!(94),
            InvalidData,
            "num_hidden_layers: 94, the size 30B-A3B holds 48",
        ),
        (
            "num_key_value_heads",
            json!(8),
            InvalidData,
            "num_key_value_heads: 8, the size 30B-A3B holds 4",
        ),
        (
            "num_attention_heads",
            json!(64),
            InvalidData,
            "num_attention_heads: 64, the size 30B-A3B holds 32",
        ),
        (
            "head_dim",
            json!(64),
            InvalidData,
            "head_dim: 64, the size 30B-A3B holds 128",
        ),
        (
            "num_experts",
            json!(64),
            InvalidData,
            "num_experts: 64, the size 30B-A3B holds 128",
        ),
        (
            "num_experts_per_tok",
            json!(4),
            InvalidData,
            "num_experts_per_tok: 4, the size 30B-A3B holds 8",
        ),
        (
            "vocab_size",
            json!(152064),
            InvalidData,
            "vocab_size: 152064, the size compiles 151936 rows",
        ),
        (
            "vocab_size",
            json!(0),
            InvalidData,
            "vocab_size: 0, the size compiles 151936 rows",
        ),
        (
            "rms_norm_eps",
            json!(1e-5),
            InvalidData,
            "rms_norm_eps: 0.00001, another than the size's",
        ),
        (
            "tie_word_embeddings",
            json!(true),
            InvalidData,
            "tie_word_embeddings: true, the size ties: false",
        ),
        (
            "max_position_embeddings",
            json!(131072),
            Unsupported,
            "max_position_embeddings: past the compiled context of 40960",
        ),
        (
            "hidden_act",
            json!("gelu"),
            Unsupported,
            "hidden_act: Qwen3 MoE's experts gate with SiLU",
        ),
        (
            "norm_topk_prob",
            json!(false),
            Unsupported,
            "norm_topk_prob: Qwen3 MoE renormalizes the selected experts' probabilities",
        ),
        (
            "mlp_only_layers",
            json!([0]),
            Unsupported,
            "mlp_only_layers: every Qwen3 MoE layer is sparse",
        ),
        (
            "decoder_sparse_step",
            json!(2),
            Unsupported,
            "decoder_sparse_step: every Qwen3 MoE layer is sparse",
        ),
        (
            "rope_scaling",
            json!({"rope_type": "linear", "factor": 2.0}),
            Unsupported,
            "found Linear { factor: 2.0 } rotating 1 of each head, compiled the unscaled spectrum over the whole head",
        ),
        (
            "use_sliding_window",
            json!(true),
            Unsupported,
            "use_sliding_window: Qwen3 MoE attends the whole history",
        ),
        (
            "attention_bias",
            json!(true),
            Unsupported,
            "attention_bias: Qwen3 MoE projects without biases",
        ),
    ];
    for (field, value, kind, reason) in cases {
        let mut document = published.clone();
        document[field] = value;
        let error = Size::<"30B-A3B">::parse(&document).map(drop).unwrap_err();
        let refusal = (error.kind(), error.to_string());
        assert_eq!(refusal, (kind, reason.to_string()), "{field}");
    }
}
