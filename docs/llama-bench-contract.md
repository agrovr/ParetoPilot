# llama-bench input format

ParetoPilot currently targets `llama.cpp` commit
[`67b9b0e7f6ce45d929a4411907d3c48ec719e81c`](https://github.com/ggml-org/llama.cpp/commit/67b9b0e7f6ce45d929a4411907d3c48ec719e81c).
Upstream does not publish a versioned JSON schema, so ParetoPilot validates a required subset and
tolerates unknown fields.

## CPU-only benchmark invocation

For an unambiguous CPU-only run, use a CPU-only build. With a multi-backend build, make CPU
intent explicit:

```bash
./llama-bench \
  -m /models/model.gguf \
  -p 512 -n 128 \
  -t 4 -b 512 -ub 128 \
  -r 10 \
  -dev none -ngl 0 -nopo 1 \
  -v \
  -o jsonl
```

Warmup is enabled unless `--no-warmup` is passed. The published study keeps warmup enabled and
records the complete command separately because JSONL does not include this fact.
Verbose mode preserves the model-buffer log marker on stderr; without `-v`, `llama-bench`
replaces the normal logger with a null callback.

The canonical candidate study pins the official Qwen 2.5 1.5B Instruct Q8_0 and Q4_0 GGUFs.
Q4_0 is required for the optimized candidates because the pinned KleidiAI integration does not
provide the same optimized path for K-quant model files.

## Parser rules

- `n_prompt > 0` and `n_gen == 0` means prompt processing (`pp`).
- `n_prompt == 0` and `n_gen > 0` means token generation (`tg`).
- Both values greater than zero means a combined test (`pg`).
- There is no upstream `test` or `repetitions` field.
- `samples_ns` and `samples_ts` must be non-empty and have equal lengths; their length is the
  recorded repetition count.
- JSONL is preferred because each completed record is flushed immediately.

`avg_ts` is the arithmetic mean of per-repetition throughput. It is not guaranteed to equal
tokens divided by `avg_ns` due to the upstream aggregation method.

## What llama-bench measures

`llama-bench` measures prompt/decode throughput. It excludes model loading, tokenization,
sampling, warmup, and server request overhead, so it does not directly measure TTFT or p95
end-to-end latency. Those metrics require a separate HTTP workload against `llama-server`.

The run manifest must additionally capture architecture, compiler and build flags, executable
checksum, model checksum, command arguments, whether the build enabled KleidiAI, and whether the
expected model-buffer marker was observed. The marker confirms what `llama.cpp` reported; it does
not prove that a particular microkernel executed. `backends` or `gpu_info` alone likewise does not
prove which backend executed the workload.

Before a comparison is accepted, ParetoPilot pools repeated artifacts per variant and requires
both summaries to agree on the build commit, model filename, workload shapes, sample counts,
synthetic status, and every declared benchmark setting except `build.kleidiai`. The generic
summary must declare that flag `false`; the optimized summary must declare it `true`.
