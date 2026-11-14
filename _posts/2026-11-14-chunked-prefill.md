---
layout: post
title: "LLM Inference #8: Chunked Prefill"
date: 2026-11-14 17:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, chunked-prefill, scheduling, vllm, latency]
description: "Three users are streaming answers. A fourth sends a long prompt, and their streams pause. How chunked prefill shares each iteration, what the token budget buys, and how vLLM implements it."
---

Three users are halfway through their answers. Tokens arrive steadily. A fourth user sends a 4,096-token document, and the streams pause while the GPU processes it.

The fourth request has no reusable prefix. The [prefix cache]({% post_url 2026-11-14-prefix-caching-kv-reuse %}) cannot help. There is room for its KV, so the [paged allocator]({% post_url 2026-11-14-paged-attention-kv-memory %}) cannot help either. The problem is how much work the scheduler puts into one iteration.

Chunked prefill splits that work across iterations. The new user waits through several chunks, while the existing users keep getting turns to decode.

## One long prompt holds up everyone else

Prefill and decode ask the GPU to do different amounts of work. A decode request usually contributes one new input token per iteration. A prefill request can contribute thousands.

With [continuous batching]({% post_url 2026-11-05-llm-inference-batching %}), those requests can share an iteration. If the scheduler includes the whole new prompt, the iteration has to process it before returning its results. The users already streaming see a longer gap between tokens.

<p class="d-sm-none" style="font-size:0.85rem;opacity:0.7;">On a narrow screen, scroll the diagrams sideways.</p>

<div role="region" aria-label="Scrollable diagram" tabindex="0" style="overflow-x:auto;">
<svg viewBox="0 0 640 262" role="img" aria-label="Three short decode iterations, then a much longer iteration containing the entire new prompt, then decoding resumes" style="width:100%;min-width:560px;height:auto;max-width:720px;">
  <defs><marker id="cp-time" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <text x="0" y="22" font-size="16" font-weight="600" fill="currentColor">The gap an existing user sees</text>
  <text x="0" y="57" font-size="14" fill="currentColor" opacity="0.75">GPU iterations</text>
  <rect x="12" y="72" width="55" height="48" rx="4" fill="#10b981"/>
  <rect x="78" y="72" width="55" height="48" rx="4" fill="#10b981"/>
  <rect x="144" y="72" width="55" height="48" rx="4" fill="#10b981"/>
  <rect x="210" y="72" width="310" height="48" rx="4" fill="#8b5cf6"/>
  <rect x="531" y="72" width="55" height="48" rx="4" fill="#10b981"/>
  <text x="365" y="92" font-size="14" fill="#ffffff" text-anchor="middle">whole prompt + decode</text>
  <text x="365" y="111" font-size="13" fill="#ffffff" text-anchor="middle">one long iteration</text>
  <text x="0" y="156" font-size="14" fill="currentColor">Tokens returned to a streaming user</text>
  <line x1="12" y1="184" x2="615" y2="184" stroke="currentColor" stroke-width="1.2" marker-end="url(#cp-time)"/>
  <circle cx="67" cy="184" r="5" fill="#10b981"/><circle cx="133" cy="184" r="5" fill="#10b981"/><circle cx="199" cy="184" r="5" fill="#10b981"/><circle cx="520" cy="184" r="5" fill="#10b981"/><circle cx="586" cy="184" r="5" fill="#10b981"/>
  <path d="M204 198 v8 H515 v-8" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="360" y="228" font-size="14" fill="#8b5cf6" text-anchor="middle">The stream waits for this iteration.</text>
  <text x="0" y="254" font-size="12" fill="currentColor" opacity="0.65">Illustrative timing. Widths are not measured GPU latencies.</text>
</svg>
</div>

You could let the running requests finish before admitting the prompt. That protects their streams, but a long answer can leave the new user waiting a long time. Giving the prompt its own full iteration has the opposite problem. The new user makes progress while everyone else waits.

The scheduler needs a way to admit some prompt work without admitting all of it.

## Give the prompt part of each iteration

Instead of processing 4,096 prompt tokens in one iteration, process a portion, keep its KV, and continue later. Put the existing decode requests in those iterations too.

This is chunked prefill. [Sarathi-Serve][sarathi] describes how to combine prefill chunks with ongoing decodes to control the throughput and latency tradeoff.

There is a throughput reason to mix them too. Prefill's many input tokens give the GPU substantial matrix work. Small decode batches spend more of their time moving weights and KV from memory. A mixed batch can use the GPU better than separate turns for each, though the gain depends on the workload.

<div role="region" aria-label="Scrollable diagram" tabindex="0" style="overflow-x:auto;">
<svg viewBox="0 0 640 290" role="img" aria-label="A whole prefill occupies one long iteration. Chunked prefill spreads the same prompt across several mixed iterations, each also including decode work" style="width:100%;min-width:560px;height:auto;max-width:720px;">
  <text x="0" y="22" font-size="16" font-weight="600" fill="currentColor">More chances to return a token</text>
  <text x="0" y="58" font-size="14" fill="currentColor">Whole prefill</text>
  <rect x="12" y="72" width="508" height="43" rx="4" fill="#8b5cf6"/>
  <rect x="12" y="115" width="508" height="13" rx="2" fill="#10b981"/>
  <text x="266" y="99" font-size="14" fill="#ffffff" text-anchor="middle">all 4,096 prompt tokens</text>
  <text x="530" y="105" font-size="13" fill="currentColor">return</text>
  <text x="0" y="164" font-size="14" fill="currentColor">Chunked prefill</text>
  <rect x="12" y="178" width="103" height="43" rx="4" fill="#8b5cf6"/><rect x="12" y="221" width="103" height="13" rx="2" fill="#10b981"/>
  <rect x="127" y="178" width="103" height="43" rx="4" fill="#8b5cf6"/><rect x="127" y="221" width="103" height="13" rx="2" fill="#10b981"/>
  <rect x="242" y="178" width="103" height="43" rx="4" fill="#8b5cf6"/><rect x="242" y="221" width="103" height="13" rx="2" fill="#10b981"/>
  <rect x="357" y="178" width="103" height="43" rx="4" fill="#8b5cf6"/><rect x="357" y="221" width="103" height="13" rx="2" fill="#10b981"/>
  <rect x="472" y="178" width="103" height="43" rx="4" fill="#8b5cf6"/><rect x="472" y="221" width="103" height="13" rx="2" fill="#10b981"/>
  <text x="64" y="205" font-size="14" fill="#ffffff" text-anchor="middle">chunk 1</text><text x="179" y="205" font-size="14" fill="#ffffff" text-anchor="middle">chunk 2</text><text x="294" y="205" font-size="14" fill="#ffffff" text-anchor="middle">chunk 3</text><text x="409" y="205" font-size="14" fill="#ffffff" text-anchor="middle">chunk 4</text><text x="524" y="205" font-size="14" fill="#ffffff" text-anchor="middle">chunk 5</text>
  <text x="64" y="254" font-size="13" fill="#10b981" text-anchor="middle">return</text><text x="179" y="254" font-size="13" fill="#10b981" text-anchor="middle">return</text><text x="294" y="254" font-size="13" fill="#10b981" text-anchor="middle">return</text><text x="409" y="254" font-size="13" fill="#10b981" text-anchor="middle">return</text><text x="524" y="254" font-size="13" fill="#10b981" text-anchor="middle">return</text>
  <text x="0" y="282" font-size="12" fill="currentColor" opacity="0.65">Purple is prefill. Green is decode. This shows scheduling, not elapsed time.</text>
</svg>
</div>

The GPU still does prefill work. Each mixed iteration can take longer than a decode-only iteration. Splitting a prompt gives the scheduler more points at which it can return decode results. It does not make the prompt's computation disappear.

The new request produces its first answer token only after the final prefill chunk. Returning between chunks helps the users already decoding.

## A 1,024-token budget, worked through

Suppose one iteration can schedule at most 1,024 input tokens. Requests A, B, and C are already decoding. Each needs one token processed to produce its next token. Request D arrives with the 4,096-token prompt.

Give A, B, and C one token each. That leaves **1,021 tokens for D**.

<div role="region" aria-label="Scrollable diagram" tabindex="0" style="overflow-x:auto;">
<svg viewBox="0 0 640 224" role="img" aria-label="An iteration's budget of 1024 tokens is divided into one token each for requests A, B, and C, leaving 1021 tokens for request D's prefill" style="width:100%;min-width:560px;height:auto;max-width:720px;">
  <text x="0" y="22" font-size="16" font-weight="600" fill="currentColor">One iteration, 1,024 input tokens</text>
  <rect x="12" y="55" width="65" height="58" rx="4" fill="#10b981"/><text x="44" y="79" font-size="15" fill="#ffffff" text-anchor="middle">A</text><text x="44" y="101" font-size="13" fill="#ffffff" text-anchor="middle">1 token</text>
  <rect x="85" y="55" width="65" height="58" rx="4" fill="#10b981"/><text x="117" y="79" font-size="15" fill="#ffffff" text-anchor="middle">B</text><text x="117" y="101" font-size="13" fill="#ffffff" text-anchor="middle">1 token</text>
  <rect x="158" y="55" width="65" height="58" rx="4" fill="#10b981"/><text x="190" y="79" font-size="15" fill="#ffffff" text-anchor="middle">C</text><text x="190" y="101" font-size="13" fill="#ffffff" text-anchor="middle">1 token</text>
  <rect x="235" y="55" width="381" height="58" rx="4" fill="#8b5cf6"/><text x="425" y="79" font-size="15" fill="#ffffff" text-anchor="middle">D</text><text x="425" y="101" font-size="14" fill="#ffffff" text-anchor="middle">1,021 prompt tokens</text>
  <text x="12" y="153" font-size="15" fill="currentColor">1 + 1 + 1 + 1,021 = 1,024</text>
  <text x="12" y="183" font-size="14" fill="currentColor">The remaining 3,075 prompt tokens wait for later iterations.</text>
  <text x="0" y="216" font-size="12" fill="currentColor" opacity="0.65">Box widths are not proportional to token counts.</text>
</svg>
</div>

Keep that order and assume A, B, and C continue streaming. No request finishes or arrives. There are no prefix-cache hits, and there is enough KV space. This example also leaves speculative decoding and the extra per-request chunk cap disabled.

| Iteration | A, B, C input tokens | D prompt tokens | D prompt tokens processed so far |
|---|---|---|---|
| 1 | 3 | 1,021 | 1,021 |
| 2 | 3 | 1,021 | 2,042 |
| 3 | 3 | 1,021 | 3,063 |
| 4 | 3 | 1,021 | 4,084 |
| 5 | 3 | 12 | 4,096 |

A, B, and C each get a decode result in every iteration. D reaches the end of its prompt in iteration 5 and can produce its first output token. In the next iteration, D becomes another ordinary one-token decode request.

The budget counts input positions processed in this iteration. It does not count the entire stored context, and it does not count output tokens returned to clients. The first iteration processes 1,024 positions but returns only three answer tokens.

## The next chunk still sees the earlier tokens

Splitting a prompt does not split it into independent documents. The later tokens still need the earlier context.

After D's first chunk, the runtime retains the KV for positions 0 through 1,020. The next chunk processes positions 1,021 through 2,041. Its attention reads the earlier KV as well as the allowed positions inside its own chunk.

<div role="region" aria-label="Scrollable diagram" tabindex="0" style="overflow-x:auto;">
<svg viewBox="0 0 640 292" role="img" aria-label="The second chunk computes queries for new prompt positions and attends to retained KV from the first chunk plus causally allowed positions in the second chunk" style="width:100%;min-width:560px;height:auto;max-width:720px;">
  <defs><marker id="cp-kv" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <text x="0" y="22" font-size="16" font-weight="600" fill="currentColor">Iteration 2 continues the same prompt</text>
  <rect x="12" y="53" width="280" height="70" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="152" y="80" font-size="15" fill="#3b82f6" text-anchor="middle">KV kept from chunk 1</text>
  <text x="152" y="105" font-size="14" fill="currentColor" text-anchor="middle">positions 0 through 1,020</text>
  <rect x="330" y="53" width="286" height="70" rx="4" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="473" y="80" font-size="15" fill="#8b5cf6" text-anchor="middle">New tokens in chunk 2</text>
  <text x="473" y="105" font-size="14" fill="currentColor" text-anchor="middle">positions 1,021 through 2,041</text>
  <path d="M152 129 V154 H310 V179" fill="none" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#cp-kv)"/>
  <path d="M473 129 V154 H330 V179" fill="none" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#cp-kv)"/>
  <rect x="126" y="185" width="390" height="47" rx="4" fill="currentColor" opacity="0.07"/>
  <text x="321" y="214" font-size="15" fill="currentColor" text-anchor="middle">Attention for the new positions</text>
  <text x="12" y="263" font-size="14" fill="currentColor">Earlier KV stays. Future positions remain masked.</text>
  <text x="12" y="285" font-size="12" fill="currentColor" opacity="0.65">The runtime computes new queries for chunk 2, not for chunk 1 again.</text>
</svg>
</div>

Causal attention makes the split possible. Earlier positions never need information from later ones. Keeping their KV lets the runtime continue without recomputing them.

There is no smaller context window here. At the end, D still has KV for the full prompt. Chunking changes when the runtime computes those positions. Different kernel shapes can introduce floating-point differences, but the intended attention calculation and context stay the same.

## Smaller chunks trade prompt progress for shorter interruptions

A smaller token budget limits how much prefill can share an iteration with decode. Existing streams usually get shorter interruptions, while the new prompt needs more iterations before its first token.

A larger budget lets the prompt make more progress per turn. It also allows more work into an iteration that existing streams must wait for. Larger chunks can use the GPU more efficiently, so cutting the budget as far as possible is rarely a good goal.

Token count is only a rough control on iteration time. Processing 1,000 new positions after a long context can cost more than processing the first 1,000 positions. Model size, batch composition, and attention kernels also matter. A 1,024-token budget is not a promise to finish within a fixed number of milliseconds.

When comparing budgets, measure time to first token for new requests and the gaps between tokens for existing streams. Include tail latency under load. An average throughput number can hide the pauses this feature is meant to reduce.

## Follow one chunk through vLLM

The walkthrough below uses vLLM V1 at [revision `a16bfee`][revision]. The links point to that revision rather than a moving branch. It follows ordinary text generation with full attention and no speculative decoding.

### The scheduler counts how much input remains

[`Request`][request] holds the prompt, output tokens, and `num_computed_tokens`. For the ordinary path, the gap between the known tokens and that counter is the work still to schedule.

For a new 4,096-token prompt, the gap starts at 4,096. After its first 1,021 positions, the gap is 3,075. During ordinary decode, the most recently sampled token has no KV yet, so the gap is one.

The same accounting can describe both phases. Chunked prefill needs no separate job for each chunk.

### The iteration has a budget, not a fixed chunk size

[`SchedulerConfig`][config] defines `max_num_batched_tokens`. The scheduler's `max_num_scheduled_tokens` defaults to that value. With speculation disabled and no separate override, the two budgets match.

[`schedule()`][schedule] visits running requests before admitting waiting requests. It limits each request's work to the budget still available. That is where our 1,021-token chunk comes from. The scheduler spends three tokens on A, B, and C, then gives D what remains.

When chunking is disabled, the [waiting-request path][waiting] does not admit a prompt whose required work exceeds the remaining budget. With chunking enabled, it can admit the portion that fits.

The line that takes that portion is:

```python
num_new_tokens = min(num_new_tokens, request_token_budget)
```

There is an optional [`long_prefill_token_threshold`][chunk-cap]. It can cap one request's contribution when several requests compete. Zero disables that extra cap. Chunk sizes can therefore change from iteration to iteration as the other requests change.

### Running does not always mean decoding

A partly prefilled request stays in the running list. On the next iteration it can compete for tokens alongside decoders.

The [optimization guide][optimization] describes the policy as prioritizing decode. The [running-request loop][running] in this revision follows list order and can encounter a partial prefill before a decoder. It does not first collect every decoder into a separate group.

Our example puts A, B, and C ahead of D and keeps that order. A different order or a crowded budget can produce a different schedule. Chunked prefill alone is not a guarantee that every stream advances on every iteration.

### KV allocation and model execution finish the job

Before admitting D, the scheduler asks [`allocate_slots`][allocate] for its KV space. [`scheduler_reserve_full_isl`][admission] defaults to true in this revision. Admission checks whether the full input can fit, even though this iteration processes only a chunk. Splitting the compute does not remove the memory requirement.

[`SchedulerOutput`][output] carries the token counts and request data to the worker. For our first iteration, the counts are:

```text
A:    1
B:    1
C:    1
D: 1021
```

The GPU runner's [`prepare_inputs`][prepare] builds the batch's token positions and sequence lengths. [`prepare_attn`][attention] supplies the attention metadata, including the block tables. D's second chunk uses a longer sequence length than its number of new tokens, because its earlier KV still belongs to the context.

<div role="region" aria-label="Scrollable diagram" tabindex="0" style="overflow-x:auto;">
<svg viewBox="0 0 640 298" role="img" aria-label="The scheduler assigns token counts, checks KV allocation, sends SchedulerOutput to the GPU runner, and advances request progress so the next iteration continues" style="width:100%;min-width:560px;height:auto;max-width:720px;">
  <defs><marker id="cp-flow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <rect x="12" y="12" width="272" height="70" rx="4" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="148" y="40" font-size="15" fill="currentColor" text-anchor="middle">schedule()</text><text x="148" y="64" font-size="14" fill="currentColor" text-anchor="middle">Assign tokens from the budget</text>
  <line x1="294" y1="47" x2="329" y2="47" stroke="currentColor" stroke-width="1.5" marker-end="url(#cp-flow)"/>
  <rect x="340" y="12" width="276" height="70" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="478" y="40" font-size="15" fill="currentColor" text-anchor="middle">allocate_slots()</text><text x="478" y="64" font-size="14" fill="currentColor" text-anchor="middle">Check and allocate KV space</text>
  <line x1="478" y1="90" x2="478" y2="121" stroke="currentColor" stroke-width="1.5" marker-end="url(#cp-flow)"/>
  <rect x="340" y="132" width="276" height="70" rx="4" fill="none" stroke="#10b981" stroke-width="1.5"/>
  <text x="478" y="160" font-size="15" fill="currentColor" text-anchor="middle">SchedulerOutput</text><text x="478" y="184" font-size="14" fill="currentColor" text-anchor="middle">Counts, request data, KV blocks</text>
  <line x1="330" y1="167" x2="295" y2="167" stroke="currentColor" stroke-width="1.5" marker-end="url(#cp-flow)"/>
  <rect x="12" y="132" width="272" height="70" rx="4" fill="none" stroke="#10b981" stroke-width="1.5"/>
  <text x="148" y="160" font-size="15" fill="currentColor" text-anchor="middle">GPU runner</text><text x="148" y="184" font-size="14" fill="currentColor" text-anchor="middle">Process this portion of the input</text>
  <text x="12" y="247" font-size="14" fill="currentColor">The scheduler records D's scheduled progress.</text>
  <text x="12" y="274" font-size="14" fill="currentColor">The next iteration continues at D's next input position.</text>
</svg>
</div>

[`_update_after_schedule`][update] advances the counter by the assigned count:

```python
request.num_computed_tokens += num_scheduled_token
```

The scheduler records this progress before GPU completion so it can prepare later work. It also tracks in-flight tokens. Treat this as scheduled progress when reading the asynchronous path.

After four 1,021-token chunks, D's counter reaches 4,084. The next turn schedules the last 12 positions. The scheduler gets chunked prefill from token counts, retained KV, and a budget applied repeatedly.

## Summary

- A whole long prefill can stretch one iteration and interrupt existing token streams.
- Chunked prefill spreads the prompt across iterations, which can also carry decode work.
- Later chunks attend to earlier KV. They continue the same prompt with the same context.
- Smaller budgets usually reduce interruptions but delay prompt completion. A token budget does not set a fixed time limit.
- vLLM tracks each request's progress and assigns work from the remaining budget. Partial prefills remain running requests, so list order matters.
- Chunking changes the compute schedule. The full prompt still needs KV space.

## References

- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024. [Paper and talk][sarathi]
- vLLM V1 source at [revision `a16bfee`][revision], linked beside the symbols used above
- [vLLM optimization guide][optimization], for the documented policy and tuning discussion

[sarathi]: https://www.usenix.org/conference/osdi24/presentation/agrawal
[revision]: https://github.com/vllm-project/vllm/tree/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576
[request]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/request.py#L63
[config]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/config/scheduler.py#L47
[chunk-cap]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/config/scheduler.py#L104
[admission]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/config/scheduler.py#L216
[schedule]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L691
[running]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L767
[waiting]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L1357
[allocate]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L1472
[output]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/output.py#L241
[prepare]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/worker/gpu/model_runner.py#L1462
[attention]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/worker/gpu/model_runner.py#L1669
[update]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L1860
[optimization]: https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill
