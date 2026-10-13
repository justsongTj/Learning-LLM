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
- SentencePiece subword tokenizer for efficient, valid UTF-8 Chinese and multilingual text;
- binary model checkpoint save/load with configuration and shape validation;
- training and autoregressive generation programs.

The implementation is intentionally educational. Matrix multiplication uses hipBLAS on DCU,
while several compound operators and their backward passes currently use host-readable loops
through the common Tensor interface. This keeps every derivative visible in C++ and makes the
CPU implementation a reference. Moving those loops into parallel HIP kernels is a later
performance milestone and does not require changing the model classes.

## Build the CPU backend

```bash
cmake -S . -B build -DTINYDL_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the demo:

```bash
./build/tinydl_demo
```

First train a SentencePiece BPE model from the same UTF-8 corpus. Byte fallback ensures that
unseen characters remain representable instead of becoming unknown tokens:

```bash
spm_train --input=corpus.txt --model_prefix=tokenizer --model_type=bpe \
  --vocab_size=8000 --character_coverage=0.9995 --byte_fallback=true
```

For a small corpus, reduce `--vocab_size` (for example to 1000). Then train TinyDL:

```bash
./build/tinydl_train corpus.txt tokenizer.model model.tdl 1000
```

Generate text from the checkpoint:

```bash
./build/tinydl_generate model.tdl tokenizer.model "你好" 100
```

## Build with DTK/DCU

Use a shell in which DTK's HIP and hipBLAS CMake packages are visible:

```bash
source /opt/dtk-26.04/env.sh
cmake -S . -B build-dcu -DTINYDL_ENABLE_DCU=ON
cmake --build build-dcu
./build-dcu/tinydl_demo --dcu
./build-dcu/tinydl_train corpus.txt tokenizer.model model.tdl 1000 --dcu
./build-dcu/tinydl_generate model.tdl tokenizer.model "你好" 100 --dcu
```

TinyDL discovers the DTK root from `ROCM_PATH`, `HIP_PATH`, or `hipconfig`, selects DTK's
underlying Clang compiler, and detects the active `gfx` architecture with `rocminfo`. On a
login node without a visible DCU, specify the architecture explicitly, for example
`-DCMAKE_HIP_ARCHITECTURES=gfx936`. Use `-DTINYDL_DTK_ROOT=/opt/dtk-26.04` only when the
environment does not expose the installation location.

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

- sequences are processed one at a time; batching is not implemented yet;
- attention is the readable quadratic implementation and has no KV cache;
- only contiguous FP32 tensors are supported;
- AdamW moment state is not included in model checkpoints;
- the SentencePiece model is stored separately and must accompany the model checkpoint;
- no distributed or mixed-precision training is included.
