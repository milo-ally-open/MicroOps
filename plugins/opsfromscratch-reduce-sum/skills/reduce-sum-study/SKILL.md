---
name: reduce-sum-study
description: Explain or review the reduce_sum operator's semantics, CPU reference, CUDA reduction structure, and Ascend C scaffold status in OpsFromScratch.
---

# Study reduce_sum

Use this workflow when the user asks how `reduce_sum` works, wants to trace its data flow, or asks for a correctness review.

1. Read `operators/reduce_sum/docs/spec.md` first and treat it as the contract.
2. Trace the CPU baseline in `operators/reduce_sum/reference/`, then compare the CUDA kernel and host wrapper with the same edge cases and tolerance.
3. State clearly that the CUDA files are unverified until built with the intended Toolkit and run on a CUDA device. The Ascend C files are placeholders and cannot be presented as an implementation.
4. For correctness reviews, check empty input, singleton input, null pointers, accumulation order, temporary allocation/lifetime, stream behavior, launch errors, and the documented tolerance.
5. Do not claim device support, test success, or performance without an actual toolchain/device run and recorded measurements.
