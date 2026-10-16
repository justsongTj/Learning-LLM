# TinyDL

TinyDL is a learning-oriented C++ implementation of the core building blocks needed by a
small GPT-style language model. It deliberately has only two backends:

- a plain, readable CPU reference backend;
- a Hygon DTK/HIP DCU backend.

The CPU backend is a correctness oracle rather than an optimized implementation. The DCU
backend uses the HIP runtime for memory and kernels and hipBLAS for matrix multiplication.

## Implemented

- contiguous FP32 tensor storage;
- backend-independent tensor operations;
- elementwise add and GELU;
- row-major matrix multiplication;
- numerically stable row softmax;
- LayerNorm;
- CPU unit tests;
- optional DTK/HIP build and a shared CPU/DCU demo.
- reverse-mode automatic differentiation with gradient accumulation;
- differentiable Linear, Embedding, GELU, LayerNorm and causal multi-head attention;
- pre-norm Transformer blocks and a decoder-only GPT model;
- mean cross-entropy loss and AdamW with global gradient clipping;
- device-resident gradient accumulation, matrix-multiplication backward, cross-entropy, and
  AdamW updates on DCU;
- SentencePiece subword tokenizer for efficient, valid UTF-8 Chinese and multilingual text;
- binary model checkpoint save/load with configuration and shape validation;
- training and autoregressive generation programs.

The implementation is intentionally educational. Matrix multiplication uses hipBLAS on DCU;
its backward pass, gradient accumulation, cross-entropy, and AdamW updates stay on the device.
Several remaining compound operators, including attention and parts of LayerNorm/embedding
backward, still use host-readable loops through the common Tensor interface. The CPU
implementation remains the correctness reference while those paths are moved to HIP kernels.

## Build the CPU backend

```bash
cmake -S . -B build -DTINYDL_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

SentencePiece 0.2.0 is vendored under `extern/sentencepiece`, so configuration and builds do
not download dependencies. Its Apache-2.0 license is preserved in that directory.

Run the demo:

```bash
./build/tinydl_demo
```

First train a SentencePiece BPE model from the same UTF-8 corpus. The repository includes
`data/corpus_zh.txt` as a small smoke-test corpus; use a much larger corpus for meaningful
language quality. Byte fallback ensures that
unseen characters remain representable instead of becoming unknown tokens:

```bash
spm_train --input=data/corpus_zh.txt --model_prefix=tokenizer --model_type=bpe \
  --vocab_size=1000 --character_coverage=0.9995 --byte_fallback=true \
  --hard_vocab_limit=false --unk_id=0 --bos_id=1 --eos_id=2 --pad_id=-1
```

BOS/EOS are required: training encodes each non-empty input line as an independent paragraph,
adds its boundary tokens, and generation stops when EOS is produced. A tokenizer trained with
`--eos_id=-1` must be retrained, and checkpoints made with the old token IDs are not compatible.

For a small corpus, reduce `--vocab_size` (for example to 1000). Then train TinyDL:

```bash
./build/tinydl_train data/corpus_zh.txt tokenizer.model model.tdl 1000
```

Training reserves the final 5% of tokens for validation, reports a rolling 100-step loss,
and uses linear warmup followed by cosine learning-rate decay. Tune the schedule with
`--learning-rate`, `--min-learning-rate`, and `--warmup-steps`; change validation frequency
with `--validation-interval`. For comparisons across changing training corpora, pass a fixed
UTF-8 validation file with `--validation-text validation.txt`; the complete positional input
file is then used for training. Every 5000 steps by default, training writes a numbered model
checkpoint plus a `.state` sidecar containing AdamW moments, optimizer/global step counts, and
the random sampler state. Change this with `--checkpoint-interval` (zero disables periodic
checkpoints). To test the optimizer and backward pass independently of corpus difficulty,
deliberately overfit one fixed sequence:

Use `--gradient-accumulation-steps N` to average gradients from `N` independently sampled
sequences before each AdamW update. The positional `steps` argument continues to count
optimizer updates, so `N=8` processes eight times as many training tokens per step while
preserving checkpoint and learning-rate schedule semantics.

```bash
./build/tinydl_train data/corpus_zh.txt tokenizer.model overfit.tdl 1000 --overfit-one-batch
```

Model dimensions are configurable at training time and stored in the checkpoint. For example:

```bash
./build/tinydl_train corpus.txt tokenizer.model model.tdl 20000 \
  --sequence-length 128 --channels 256 --heads 4 --layers 6 --ffn-channels 1024
```

Resume a new-format checkpoint for an additional number of steps. Model dimensions are read
from the checkpoint, while the learning-rate schedule applies to this continuation run:

```bash
./build/tinydl_train corpus.txt tokenizer.model continued.tdl 10000 \
  --resume model.tdl --learning-rate 3e-5 --min-learning-rate 1e-5 --warmup-steps 0
```

Old model-only checkpoints have no optimizer sidecar. They can still be continued once by
providing their known completed step; AdamW starts fresh, and subsequent checkpoints contain
the complete state:

```bash
./build/tinydl_train corpus.txt tokenizer.model continued.tdl 10000 \
  --resume old-model.tdl --resume-step 20000 \
  --learning-rate 3e-5 --min-learning-rate 1e-5 --warmup-steps 0
```

Generate text from the checkpoint:

```bash
./build/tinydl_generate model.tdl tokenizer.model "你好" 100
```

Sampling defaults to temperature 0.8, top-k 40, and a 1.1 repetition penalty over the most
recent 64 tokens. Consecutive standalone whitespace tokens are suppressed. Greedy decoding is
useful for inspecting the model's most likely continuation, while a fixed seed makes sampling
reproducible:

```bash
./build/tinydl_generate model.tdl tokenizer.model "你好" 100 --temperature 0.1 --top-k 1
./build/tinydl_generate model.tdl tokenizer.model "你好" 100 --seed 42
./build/tinydl_generate model.tdl tokenizer.model "你好" 100 \
  --repetition-penalty 1.15 --repetition-window 64 --min-new-tokens 20
```

`--min-new-tokens` temporarily excludes EOS from sampling. It controls premature endings but
does not improve model knowledge, so keep it well below the maximum generated-token count.

## Prepare a larger corpus

`scripts/prepare_hf_corpus.py` streams a configurable Hugging Face dataset, normalizes Unicode,
filters very short/non-Chinese lines, removes exact duplicates, and makes a deterministic
train/validation split. Streaming avoids downloading the entire source dataset first. Install
the optional data dependency and build a 200 MiB mixture of the existing corpus and Chinese
Wikipedia:

```bash
python3 -m pip install --user datasets

python3 scripts/prepare_hf_corpus.py \
  --local-text "$HOME/tujie/apps/datasets/corpus_wikipedia_zh_15mb.txt" \
  --dataset wikimedia/wikipedia \
  --config 20231101.zh \
  --split train \
  --text-column text \
  --max-output-mb 200 \
  --train-output "$HOME/tujie/apps/datasets/corpus_zh_200mb_train.txt" \
  --validation-output "$HOME/tujie/apps/datasets/corpus_zh_fixed_validation.txt"
```

The dataset, configuration, split, and text column are command-line parameters because Hub
datasets can use different schemas. Inspect the dataset card before changing them. Keep the
generated validation file unchanged across experiments, and continue an existing model with
the same tokenizer:

For servers that can reach ModelScope but not Hugging Face, the same cleaner can stream a
ModelScope dataset. Multiple fields can be joined into one training paragraph:

```bash
python3 -m pip install --user modelscope addict
python3 scripts/prepare_hf_corpus.py \
  --local-text "$HOME/tujie/apps/datasets/corpus_wikipedia_zh_15mb.txt" \
  --modelscope-dataset AI-ModelScope/train_1M_CN \
  --split train \
  --text-columns instruction,input,output \
  --instruction-format \
  --max-output-mb 200 \
  --train-output "$HOME/tujie/apps/datasets/corpus_zh_200mb_train.txt" \
  --validation-output "$HOME/tujie/apps/datasets/corpus_zh_fixed_validation.txt"
```

Outputs are first written as `.part` files and renamed only after successful completion, so a
network failure cannot leave an incomplete file under the requested final name.

```bash
./build/tinydl_train \
  "$HOME/tujie/apps/datasets/corpus_zh_200mb_train.txt" \
  "$HOME/tujie/apps/tinydl-artifacts/tokenizer_zh_8k_eos.model" \
  "$HOME/tujie/apps/tinydl-artifacts/model-zh-expanded.tdl" \
  20000 \
  --validation-text "$HOME/tujie/apps/datasets/corpus_zh_fixed_validation.txt" \
  --resume "$HOME/tujie/apps/tinydl-artifacts/model-wiki-zh-8k-eos-30k.tdl" \
  --learning-rate 3e-5 --min-learning-rate 1e-5 --warmup-steps 0 \
  --gradient-accumulation-steps 8 \
  --validation-interval 500 --checkpoint-interval 2500 --dcu
```

## Build with DTK/DCU

Use a shell in which DTK's HIP and hipBLAS CMake packages are visible:

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B build-dcu
cmake --build build-dcu
./build-dcu/tinydl_demo --dcu
./build-dcu/tinydl_train corpus.txt tokenizer.model model.tdl 1000 --dcu
./build-dcu/tinydl_generate model.tdl tokenizer.model "你好" 100 --dcu
```

When `ROCM_PATH` is present, TinyDL enables both CPU and DCU backends by default. It uses
that root to select
DTK's underlying Clang compiler and find HIP packages, then detects the active `gfx`
architecture with `rocminfo`. If DCU is explicitly enabled while `ROCM_PATH` is unset or
invalid, configuration stops with a clear error. On a login node without a visible DCU,
specify the architecture explicitly, for example `-DCMAKE_HIP_ARCHITECTURES=gfx936`.

If a particular DTK release exports a different hipBLAS CMake target name, change only the
target name in `CMakeLists.txt`; the C++ backend interface does not change.

## Design rules

1. Every new operation gets a plain CPU implementation first.
2. CPU results are the reference for the DCU implementation.
3. No SIMD, OpenMP or third CPU backend is added.
4. Model code never directly calls HIP or hipBLAS.
5. FP32 correctness comes before FP16 or kernel fusion.

## Source map

- `autograd.cpp`: graph construction, reverse traversal and derivative formulas;
- `nn.cpp`: Linear, Embedding, Attention, Transformer and GPT composition;
- `optimizer.cpp`: AdamW update and gradient clipping;
- `tokenizer.cpp`: byte-level encoding and decoding;
- `checkpoint.cpp`: versioned binary weight format;
- `apps/train.cpp`: next-token training loop;
- `apps/generate.cpp`: autoregressive top-k sampling loop.

## Deliberate limitations

- physical tensor batching is not implemented; gradient accumulation provides a configurable
  effective batch across independently sampled sequences;
- attention is the readable quadratic implementation and has no KV cache;
- only contiguous FP32 tensors are supported;
- AdamW moment state lives in the `.tdl.state` sidecar rather than the model file;
- the SentencePiece model is stored separately and must accompany the model checkpoint;
- no distributed or mixed-precision training is included.
