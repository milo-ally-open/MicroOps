# Storyboard quality guide

Use this when a video needs a stronger explanatory arc than a short code walkthrough. The target is original, intuitive mathematical storytelling with polished motion graphics. Do not imitate another creator's signature compositions, wording, or branding.

## Story spine

1. **Question:** Open with the one idea the viewer should understand, not a list of learning objectives.
2. **Intuition:** Start from a tiny, concrete vector, matrix, number line, or curve. Label toy values and make the key relationship visible before showing implementation details.
3. **Reveal:** Animate the transformation that resolves the question. Reuse the same objects as they become equation symbols, graph samples, tensor entries, or thread-owned values.
4. **Hardware mapping:** Explain which logical work item owns each value, what it reads, computes, and writes, and how the host launches the kernel. Separate logical mapping from physical execution order.
5. **Takeaway:** Return to the opening question and state one short correctness invariant or mental model.

The story spine is a guide, not a requirement for five separate scenes. Merge beats when an uninterrupted transformation is clearer.

## Mathematics should drive the visual metaphor

Choose the simplest visual that makes the operator's relation intuitive, then verify it against the spec and source:

| Operator family | Useful starting intuition | Avoid implying |
| --- | --- | --- |
| Add | Aligned contributions flow into the same output index | Inputs at different indices mix |
| Subtract | Subtracting a value reverses its displacement | Scalar subtraction is a reduction |
| Multiply | Signed scale and per-index products | An area metaphor is the implementation itself |
| Divide | A quotient answers how many denominator-sized units fit | Zero-denominator behavior is specially handled when it is not |
| ReLU | A hinge clips the negative half-line to zero | Negative values pass through |
| GELU | A smooth, input-dependent gate modulates the signal | The exact erf form when the kernel uses the tanh approximation |
| SiLU | The signal is multiplied by a smooth sigmoid gate | The gate is binary |
| Sigmoid | A logit is mapped to a bounded probability-like value | The kernel must compute an unstable textbook expression verbatim |
| Tanh | An odd, centered curve saturates toward −1 and +1 | Its outputs reach those limits at finite inputs |

When plotting a function, evaluate the selected implementation, label axes and range, and use a few samples whose displayed outputs match the curve. If a kernel contains a stability branch or approximation, show the smooth mathematical curve and then explain the implementation detail as a separate but connected beat.

## Motion and composition

- Give every animation a clear focal point. Change one important visual relationship at a time.
- Move, split, combine, reveal, or trace objects when that motion represents the computation. Use color consistently for operands, decisions, and results.
- Keep prose sparse. Put longer explanations in the operator's `docs/usage.md`; use narration only when requested.
- Use consistent margins and readable type at the delivery resolution. Check mathematical labels, code, CJK glyphs, color contrast, and graph scaling in actual rendered frames.
- Avoid generic GPU glamour shots, unrelated transitions, decorative particle effects, and static slides that merely restate code.

## Hardware truth gate

- Inspect the selected variant's host and kernel before choosing the index mapping, block geometry, values per thread, tail behavior, or memory levels to draw.
- Draw a small subset of workers in detail and mark omitted workers with an ellipsis. Do not resize hardware boxes in a way that claims to be to scale.
- Label conceptual lane/SM diagrams as conceptual. A block's SM assignment can be shown as a relationship; block ordering, concurrency, occupancy, exact memory transactions, and captured runtime behavior require separate evidence.
- Show only memory spaces and synchronization used by the implementation. Never add shared memory, a reduction tree, cache hits, or device transfers as generic decoration.
- A source or rendering review is not a correctness test, device test, memory measurement, or benchmark.

## Per-operator review and correction loop

Run this gate for every operator in a batch, even when the scenes inherit from the same animation helper. Shared code can reduce layout drift, but it can also propagate a bad label or make every operator look interchangeable. Review each exported video and its own source separately.

### 1. Freeze the evidence for the scene

- Read that operator's spec, selected kernel, host launch, and scene source. Record the exact variant and backend being taught.
- Trace every visible formula, branch, launch parameter, index, memory-space label, and boundary condition to those files. Mark the diagram conceptual wherever it explains a model rather than reports a device observation.
- Choose a tiny, deterministic example that exercises the idea. Keep its inputs and outputs in one source of truth where practical, then check every displayed result independently (a short Python calculation is enough). For floating-point functions, label rounded outputs as approximate and check they agree with the plotted implementation within the shown precision.
- Include a boundary example only when it reflects the actual host/kernel contract. For a tail, calculate which indices are valid, which workers are launched, and what the guard prevents. Do not imply guarded threads performed loads or stores.

### 2. Check that the math story belongs to this operator

- State one learning question, then make the chosen visual answer it. Reusing a color palette or hardware chapter is fine; swapping only the operator name in an otherwise identical story is not.
- For binary operations, follow both same-index operands into the result. For activations, evaluate and plot the implementation actually used, connect sample markers to the matching sample table, and explain approximations or stability branches without suggesting they change the mathematical function.
- Check signs, parentheses, units, shapes, index labels, and rounding. Pay extra attention to examples involving negative values, fractions, thresholds, asymptotes, NaN/Inf, and zero denominators because these expose mismatches that a happy-path example can hide.

### 3. Check the hardware explanation against the launch

- Derive logical work ownership from the host launch and kernel index expression. Verify the displayed block/tile sizes and the first, last, and tail indices arithmetically.
- Draw only the movement and storage present in that implementation. Connect input loads, per-worker computation, and output stores to the correct indices; for binary operators, keep both inputs on the same logical worker.
- Keep the distinction visible between logical workers and physical execution. A lane diagram does not prove a fixed schedule, simultaneous execution, SM/Core assignment, occupancy, memory transaction count, or performance. Label the walkthrough conceptual unless it came from an actual, identified profile.

### 4. Review the rendered artifact, not just the scene code

- Run `info` and full-stream `audit` for each candidate. Require the requested resolution and frame rate; treat audit as a decode/metadata check only.
- Inspect a contact sheet for the complete arc, then inspect full-resolution frames at the opening, each dense or changing section, the hardware mapping, the boundary case, and the ending. If the contact sheet reveals a questionable transition, extract adjacent timestamps and inspect those too.
- At delivery resolution, check legibility, margins, overlaps, glyph fallback, line breaks, graph labels, marker/table agreement, transition remnants, pacing, and that the final frame resolves cleanly. A readable preview is not enough for a 720p deliverable.
- Keep a per-operator verdict while reviewing the batch: source/math fidelity, hardware fidelity, visual/layout quality, and render integrity. A defect in any one category means revise that scene or shared helper, render it to a new candidate ID, and repeat the affected checks. Recheck all operators if a shared helper changed.

### 5. Preserve review evidence and deliver deliberately

- Keep candidates, contact sheets, extracted frames, and review notes under ignored `build/manim/`. Use a new candidate ID for every changed render; never overwrite a candidate or delete older videos as a shortcut.
- Do not describe an unrendered or uninspected scene as complete. Report which operators were reviewed, the concrete defects fixed, what metadata/decode checks passed, and anything that remains unverified.
- Move only the explicitly selected/approved candidate into `operators/<op>/animation/videos/`. A render passing technical checks does not itself mean it is approved for publication.
