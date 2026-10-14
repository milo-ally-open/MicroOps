---
name: add-operator
description: Add a new operator to OpsFromScratch with aligned semantics, backend status, tests, docs, examples, and packaging metadata.
---

# Add an operator

Use this workflow when adding a new operator to this repository.

1. Read `docs/architecture.md`, the closest existing operator's `docs/spec.md`, and the root `CMakeLists.txt` before editing.
2. Create `operators/<operator>/` with `docs/{proposal,spec,tasks,usage}.md`, `operator.yaml`, a CPU reference, backend directories, tests, benchmarks, examples, and `animation/` when those items apply.
3. Define input/output semantics, dtypes, shapes, empty/boundary behavior, errors, precision tolerance, and acceptance checks in `spec.md` before implementing kernels.
4. Make the CPU reference the semantic baseline. Keep each optional backend behind its own disabled-by-default CMake option; CPU-only configuration must not probe for GPU/NPU SDKs.
5. Mark a backend available only after a real toolchain build and target-device correctness run. Otherwise mark it scaffold or unverified and make build/test commands explain why they cannot run.
6. Add CPU tests for normal and boundary inputs. Device tests must report missing compiler, SDK, or hardware as skipped/unavailable rather than passed.
7. Add build, test, benchmark, and package wiring. Benchmark rows must come from actual runs and retain device, compiler/runtime, input size, and measurements.
8. Update the root README and operator metadata, then run checks that the current environment supports. Report checks that could not run and their concrete reason.
