---
layout: post
title: "Batching: Getting More Tokens from the Same Bytes"
date: 2026-11-05 00:00:00 -0400
categories: [AI, LLM Inference]
tags: [llm, inference, batching, continuous-batching, kv-cache, scheduling]
description: "Why batching makes LLM decoding cheaper, why static batching wastes most of the gain, and how continuous batching fixes it, explained with diagrams instead of code."
---

Earlier in this series we looked at what one decode step actually does: to produce a single next token, the GPU reads every model weight out of memory, plus that sequence's KV cache. The arithmetic done per byte read is tiny. At batch size 1, the GPU spends most of each step waiting on memory, not computing. (See the [KV cache, prefill, and decode article]({% post_url 2026-09-23-llm-inference-kv-cache-prefill-decode %}) for that background.)

That leaves an obvious question. If we are paying to read all those bytes anyway, can we get more out of each read?

That is what batching does. This post covers why it works, why the simple version throws away most of the gain, how **continuous batching** recovers it, and the two new problems the fix creates.

## Why batching works

Take a 7-billion-parameter model stored in 16-bit floats. That is 7B × 2 bytes ≈ 14 GB of weights, and every decode step reads all of it.

At batch size 1, those 14 GB produce exactly one token. The work per weight is one multiply and one add, so the compute units finish almost immediately and then sit waiting for the next bytes to arrive. Decode at batch 1 is **memory-bandwidth bound**: the speed limit is how fast bytes move, not how fast the GPU computes.

Now run 32 sequences through the same step. Each weight is read once and used 32 times, once per sequence. The step still reads about 14 GB, but it now produces 32 tokens. Since the step was waiting on memory anyway, the extra arithmetic is close to free.

<svg viewBox="0 0 720 210" role="img" aria-label="A decode step at batch 1 and batch 32 reads the same 14 GB of weights, but produces 1 token versus 32 tokens" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="arrA" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <rect x="110" y="10" width="12" height="12" rx="2" fill="#3b82f6"/>
  <text x="128" y="20" font-size="12" fill="currentColor">memory traffic</text>
  <rect x="242" y="10" width="12" height="12" rx="2" fill="#f59e0b"/>
  <text x="260" y="20" font-size="12" fill="currentColor">arithmetic</text>
  <text x="0" y="62" font-size="13" font-weight="600" fill="currentColor">batch = 1</text>
  <rect x="110" y="44" width="360" height="24" rx="3" fill="#3b82f6"/>
  <text x="290" y="61" font-size="12" fill="#ffffff" text-anchor="middle">read 14 GB of weights</text>
  <rect x="110" y="72" width="24" height="18" rx="3" fill="#f59e0b"/>
  <text x="142" y="86" font-size="11" fill="currentColor">compute units mostly idle</text>
  <line x1="482" y1="56" x2="518" y2="56" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrA)"/>
  <text x="526" y="61" font-size="13" font-weight="600" fill="currentColor">1 token</text>
  <text x="0" y="140" font-size="13" font-weight="600" fill="currentColor">batch = 32</text>
  <rect x="110" y="122" width="360" height="24" rx="3" fill="#3b82f6"/>
  <text x="290" y="139" font-size="12" fill="#ffffff" text-anchor="middle">read the same 14 GB of weights</text>
  <rect x="110" y="150" width="300" height="18" rx="3" fill="#f59e0b"/>
  <text x="418" y="164" font-size="11" fill="currentColor">compute units busy</text>
  <line x1="482" y1="134" x2="518" y2="134" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrA)"/>
  <text x="526" y="139" font-size="13" font-weight="600" fill="currentColor">32 tokens</text>
  <text x="110" y="196" font-size="12" fill="currentColor" opacity="0.75">Step time is set by the blue bar. Widening the amber bar underneath it is nearly free.</text>
</svg>

That is the whole trick: **same bytes, more tokens.**

### A systems analogy: group commit

If you have worked on databases, this is group commit. One `fsync` is expensive, so you collect many transactions and pay for one `fsync` on behalf of all of them. Two properties carry over directly:

- Throughput rises much faster than per-request cost.
- Waiting to fill a batch adds latency for whoever arrived first.

**Where the analogy breaks.** In group commit, the transactions in a batch share the `fsync` and add almost nothing else. In LLM decoding, each sequence brings its own KV cache. It is shared with nobody, it has to be read every step, and it grows by one token's worth of keys and values every step. The weights get amortized across the batch; the KV cache reads do not.

<svg viewBox="0 0 720 250" role="img" aria-label="Weights are shared across the batch while each sequence carries its own growing KV cache" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="arrB" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <rect x="150" y="16" width="420" height="34" rx="4" fill="#3b82f6"/>
  <text x="360" y="38" font-size="12.5" fill="#ffffff" text-anchor="middle">model weights, 14 GB — read once per step, used by the whole batch</text>
  <line x1="205" y1="50" x2="205" y2="78" stroke="currentColor" stroke-width="1.3" marker-end="url(#arrB)"/>
  <line x1="320" y1="50" x2="320" y2="78" stroke="currentColor" stroke-width="1.3" marker-end="url(#arrB)"/>
  <line x1="435" y1="50" x2="435" y2="78" stroke="currentColor" stroke-width="1.3" marker-end="url(#arrB)"/>
  <line x1="550" y1="50" x2="550" y2="78" stroke="currentColor" stroke-width="1.3" marker-end="url(#arrB)"/>
  <rect x="160" y="82" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="205" y="97" font-size="11.5" fill="currentColor" text-anchor="middle">sequence 1</text>
  <rect x="275" y="82" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="320" y="97" font-size="11.5" fill="currentColor" text-anchor="middle">sequence 2</text>
  <rect x="390" y="82" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="435" y="97" font-size="11.5" fill="currentColor" text-anchor="middle">sequence 3</text>
  <rect x="505" y="82" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="550" y="97" font-size="11.5" fill="currentColor" text-anchor="middle">sequence 4</text>
  <rect x="175" y="112" width="60" height="26" rx="3" fill="#10b981"/>
  <rect x="290" y="112" width="60" height="64" rx="3" fill="#10b981"/>
  <rect x="405" y="112" width="60" height="40" rx="3" fill="#10b981"/>
  <rect x="520" y="112" width="60" height="88" rx="3" fill="#10b981"/>
  <text x="205" y="152" font-size="11" fill="currentColor" text-anchor="middle">KV cache</text>
  <text x="320" y="190" font-size="11" fill="currentColor" text-anchor="middle">KV cache</text>
  <text x="435" y="166" font-size="11" fill="currentColor" text-anchor="middle">KV cache</text>
  <text x="550" y="214" font-size="11" fill="currentColor" text-anchor="middle">KV cache</text>
  <text x="150" y="238" font-size="12" fill="currentColor" opacity="0.75">Each KV cache is private, sized by that request's conversation, and one token taller after every step.</text>
</svg>

So adding a sequence to the batch costs real memory, and that cost keeps growing while the sequence runs.

## Why you cannot batch forever

Two limits show up as the batch grows.

**Compute catches up.** The arithmetic per step grows with batch size while the weight bytes stay fixed. At some batch size the GPU stops waiting on memory and starts waiting on its own compute units. Past that crossover a bigger batch is no longer nearly free: each step takes longer, so every request sees slower token-to-token latency.

**KV cache memory runs out.** Every sequence in the batch needs its KV cache resident in GPU memory, right next to the weights. In practice this is usually the wall you hit first, and it shapes the rest of this post.

<svg viewBox="0 0 720 300" role="img" aria-label="Throughput rises steeply with batch size then flattens; KV cache memory usually runs out before the compute crossover" style="width:100%;height:auto;max-width:720px;">
  <rect x="427.5" y="40" width="222.5" height="210" fill="currentColor" opacity="0.07"/>
  <line x1="90" y1="250" x2="660" y2="250" stroke="currentColor" stroke-width="1.2"/>
  <line x1="90" y1="40" x2="90" y2="250" stroke="currentColor" stroke-width="1.2"/>
  <polyline points="90,243.5 157.5,237.5 225,226.3 292.5,207.3 360,178.9 427.5,143.3 495,107.8 562.5,79.3 630,60.4" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
  <line x1="427.5" y1="40" x2="427.5" y2="250" stroke="#ef4444" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="434" y="54" font-size="11.5" fill="#ef4444" font-weight="600">KV cache memory runs out</text>
  <text x="434" y="69" font-size="11" fill="currentColor" opacity="0.8">everything to the right is unreachable</text>
  <line x1="562.5" y1="90" x2="562.5" y2="250" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.6"/>
  <text x="569" y="104" font-size="11" fill="currentColor" opacity="0.8">compute catches up</text>
  <text x="569" y="118" font-size="11" fill="currentColor" opacity="0.8">(steps get slower)</text>
  <text x="140" y="120" font-size="11.5" fill="currentColor" opacity="0.85">nearly free: each added</text>
  <text x="140" y="135" font-size="11.5" fill="currentColor" opacity="0.85">sequence rides the same bytes</text>
  <line x1="215" y1="142" x2="262" y2="200" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <text x="90" y="268" font-size="10" fill="currentColor" text-anchor="middle">1</text>
  <text x="157.5" y="268" font-size="10" fill="currentColor" text-anchor="middle">2</text>
  <text x="225" y="268" font-size="10" fill="currentColor" text-anchor="middle">4</text>
  <text x="292.5" y="268" font-size="10" fill="currentColor" text-anchor="middle">8</text>
  <text x="360" y="268" font-size="10" fill="currentColor" text-anchor="middle">16</text>
  <text x="427.5" y="268" font-size="10" fill="currentColor" text-anchor="middle">32</text>
  <text x="495" y="268" font-size="10" fill="currentColor" text-anchor="middle">64</text>
  <text x="562.5" y="268" font-size="10" fill="currentColor" text-anchor="middle">128</text>
  <text x="630" y="268" font-size="10" fill="currentColor" text-anchor="middle">256</text>
  <text x="375" y="290" font-size="12" fill="currentColor" text-anchor="middle">batch size</text>
  <text x="-145" y="24" transform="rotate(-90)" font-size="12" fill="currentColor" text-anchor="middle">throughput (tokens/s)</text>
  <text x="90" y="34" font-size="11" fill="currentColor" opacity="0.6">shape is illustrative, not a benchmark</text>
</svg>

## Static batching and its waste

The simplest way to batch: collect N requests, run them together, return the results when the batch is done.

The catch hides in "when the batch is done." A batch finishes when its **longest** sequence finishes, and output lengths vary enormously, from a one-line answer to a multi-page essay, with no way to know in advance which one you are getting.

Say one request in the batch needs 20 tokens and another needs 2,000. The short one finishes at step 20, and its slot then sits empty for 1,980 steps. Meanwhile new requests wait in the queue, because the batch is not done.

If you think of the batch as a set of parallel workers, static batching is a **bulk-synchronous job with a barrier**: nobody moves on until the slowest worker arrives.

<svg viewBox="0 0 720 230" role="img" aria-label="Static batching timeline: four slots, requests A to D, with slots idle until the longest request finishes at the barrier" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="22" font-size="13" font-weight="600" fill="currentColor">Static batching — 8 requests, 4 slots</text>
  <text x="120" y="42" font-size="10" fill="currentColor" opacity="0.6">step 1</text>
  <text x="408" y="42" font-size="10" fill="currentColor" opacity="0.6">step 9</text>
  <text x="0" y="68" font-size="11.5" fill="currentColor">slot 1</text>
  <rect x="120" y="52" width="108" height="22" rx="2" fill="#3b82f6"/><text x="174" y="67" font-size="11" fill="#ffffff" text-anchor="middle">A</text>
  <rect x="228" y="52" width="180" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="318" y="67" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">idle</text>
  <rect x="408" y="52" width="144" height="22" rx="2" fill="#ef4444"/><text x="480" y="67" font-size="11" fill="#ffffff" text-anchor="middle">E</text>
  <rect x="552" y="52" width="72" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <text x="0" y="98" font-size="11.5" fill="currentColor">slot 2</text>
  <rect x="120" y="82" width="288" height="22" rx="2" fill="#6366f1"/><text x="264" y="97" font-size="11" fill="#ffffff" text-anchor="middle">B — the longest in its group</text>
  <rect x="408" y="82" width="72" height="22" rx="2" fill="#14b8a6"/><text x="444" y="97" font-size="11" fill="#ffffff" text-anchor="middle">F</text>
  <rect x="480" y="82" width="144" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="552" y="97" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">idle</text>
  <text x="0" y="128" font-size="11.5" fill="currentColor">slot 3</text>
  <rect x="120" y="112" width="72" height="22" rx="2" fill="#10b981"/><text x="156" y="127" font-size="11" fill="#ffffff" text-anchor="middle">C</text>
  <rect x="192" y="112" width="216" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="300" y="127" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">idle</text>
  <rect x="408" y="112" width="216" height="22" rx="2" fill="#8b5cf6"/><text x="516" y="127" font-size="11" fill="#ffffff" text-anchor="middle">G</text>
  <text x="0" y="158" font-size="11.5" fill="currentColor">slot 4</text>
  <rect x="120" y="142" width="180" height="22" rx="2" fill="#f59e0b"/><text x="210" y="157" font-size="11" fill="#ffffff" text-anchor="middle">D</text>
  <rect x="300" y="142" width="108" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="354" y="157" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">idle</text>
  <rect x="408" y="142" width="108" height="22" rx="2" fill="#ec4899"/><text x="462" y="157" font-size="11" fill="#ffffff" text-anchor="middle">H</text>
  <rect x="516" y="142" width="108" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <line x1="408" y1="46" x2="408" y2="172" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text x="414" y="188" font-size="11.5" fill="#ef4444" font-weight="600">barrier: E–H cannot start until A–D have all finished</text>
  <text x="0" y="214" font-size="12" fill="currentColor" opacity="0.8">14 steps. 23 of the 56 slot-steps the GPU offered went to empty seats.</text>
</svg>

## Continuous batching

The fix is to stop treating the batch as a fixed unit. Instead, the scheduler decides batch membership **every step**:

- When a sequence emits its end token, it leaves the batch immediately.
- On the next step, a waiting request takes the free slot.

This is **continuous batching**, also called iteration-level scheduling. It was introduced in the Orca paper at OSDI 2022 ([Yu et al.][orca]) and is now standard in serving engines such as vLLM.

In systems terms, the barrier becomes a **work queue**: each worker grabs the next task the moment it is free.

<svg viewBox="0 0 720 220" role="img" aria-label="Continuous batching timeline: the same eight requests finish in 11 steps because free slots are refilled immediately" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="22" font-size="13" font-weight="600" fill="currentColor">Continuous batching — the same 8 requests, the same 4 slots</text>
  <text x="120" y="42" font-size="10" fill="currentColor" opacity="0.6">step 1</text>
  <text x="0" y="68" font-size="11.5" fill="currentColor">slot 1</text>
  <rect x="120" y="52" width="108" height="22" rx="2" fill="#3b82f6"/><text x="174" y="67" font-size="11" fill="#ffffff" text-anchor="middle">A</text>
  <rect x="228" y="52" width="72" height="22" rx="2" fill="#14b8a6"/><text x="264" y="67" font-size="11" fill="#ffffff" text-anchor="middle">F</text>
  <rect x="300" y="52" width="216" height="22" rx="2" fill="#8b5cf6"/><text x="408" y="67" font-size="11" fill="#ffffff" text-anchor="middle">G</text>
  <text x="0" y="98" font-size="11.5" fill="currentColor">slot 2</text>
  <rect x="120" y="82" width="288" height="22" rx="2" fill="#6366f1"/><text x="264" y="97" font-size="11" fill="#ffffff" text-anchor="middle">B</text>
  <rect x="408" y="82" width="108" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="462" y="97" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">queue empty</text>
  <text x="0" y="128" font-size="11.5" fill="currentColor">slot 3</text>
  <rect x="120" y="112" width="72" height="22" rx="2" fill="#10b981"/><text x="156" y="127" font-size="11" fill="#ffffff" text-anchor="middle">C</text>
  <rect x="192" y="112" width="144" height="22" rx="2" fill="#ef4444"/><text x="264" y="127" font-size="11" fill="#ffffff" text-anchor="middle">E</text>
  <rect x="336" y="112" width="180" height="22" rx="2" fill="currentColor" opacity="0.1"/><text x="426" y="127" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.7">queue empty</text>
  <text x="0" y="158" font-size="11.5" fill="currentColor">slot 4</text>
  <rect x="120" y="142" width="180" height="22" rx="2" fill="#f59e0b"/><text x="210" y="157" font-size="11" fill="#ffffff" text-anchor="middle">D</text>
  <rect x="300" y="142" width="108" height="22" rx="2" fill="#ec4899"/><text x="354" y="157" font-size="11" fill="#ffffff" text-anchor="middle">H</text>
  <rect x="408" y="142" width="108" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <text x="0" y="186" font-size="12" fill="#10b981">A free seat is refilled on the very next step. Watch slot 3: C ends, E moves straight in.</text>
  <text x="0" y="208" font-size="12" fill="currentColor" opacity="0.8">11 steps instead of 14. The only idle time left is the tail, when nothing is waiting.</text>
</svg>

**Where this analogy breaks.** Workers pulling from a queue are interchangeable. Batch slots are not. A sequence with a long prompt and a long conversation carries a far larger KV cache than a short one. So "is there a free slot?" is not really a slot-count question. It is a memory-allocation question: is there room for *this* request's KV cache, now and as it grows?

## What the difference looks like at scale

The scheduling effect is easy to isolate without a GPU. Give 1,000 requests a random output length between 10 and 1,000 tokens, run them through both schedulers with the same batch size, and count how many GPU steps each needs. Every step advances every active request by one token, so both schedulers generate exactly the same tokens; only the seating differs.

| Batch size | Output lengths | Static | Continuous | Speed-up |
|---|---|---|---|---|
| 16 | 10–1,000 | 59,603 steps, 53% of slots busy | 32,313 steps, 98% of slots busy | 1.84× |
| 64 | 10–1,000 | 15,726 steps, 50% of slots busy | 8,555 steps, 93% of slots busy | 1.84× |
| 16 | all exactly 500 | 31,500 steps, 99% of slots busy | 31,500 steps, 99% of slots busy | 1.00× |

Two things fall out of that table.

**The waste comes from variance, not from batching.** When every request produces exactly 500 tokens, the two schedulers tie perfectly. It is the mix of short and long outputs that leaves static batching's seats empty.

**Bigger static batches are worse, not better.** Going from 16 slots to 64 drops static utilization from 53% to 50%, because a larger group is more likely to contain one very long request that everyone else waits on.

This model deliberately ignores two things: the cost of processing prompts, and memory limits. Every slot is treated as identical and free. Those two omissions are exactly the next section.

## The new problems continuous batching creates

### Prefill stalls

Before a new request can generate anything, it has to process its whole prompt. This phase is **prefill**. Unlike decode, prefill handles all the prompt's tokens at once, so it is compute-heavy rather than memory-bound.

With continuous batching, a new request joins mid-flight, inside the same step as everyone else's decode. If it brings a long prompt, that one step takes far longer, and every request already running waits for it. Users see it as a stall in the middle of a stream of tokens.

<svg viewBox="0 0 720 240" role="img" aria-label="Without chunking one step balloons to absorb a long prompt; chunked prefill spreads the prompt across several steps of similar size" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="20" font-size="12.5" font-weight="600" fill="currentColor">One long prompt joins the batch</text>
  <text x="0" y="56" font-size="11.5" fill="currentColor">plain</text>
  <rect x="110" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="160" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="210" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="260" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="310" y="40" width="240" height="26" rx="2" fill="#ef4444"/>
  <text x="430" y="58" font-size="11.5" fill="#ffffff" text-anchor="middle">decode + a 4,000-token prefill</text>
  <rect x="552" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="602" y="40" width="48" height="26" rx="2" fill="#3b82f6"/>
  <path d="M310,34 L310,28 L550,28 L550,34" fill="none" stroke="#ef4444" stroke-width="1.3"/>
  <text x="430" y="22" font-size="11" fill="#ef4444" text-anchor="middle">every active request waits out this one step</text>
  <text x="110" y="86" font-size="11" fill="currentColor" opacity="0.7">each small box is one decode step for the whole batch</text>
  <text x="0" y="134" font-size="12.5" font-weight="600" fill="currentColor">Chunked prefill</text>
  <text x="0" y="170" font-size="11.5" fill="currentColor">chunked</text>
  <rect x="110" y="154" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="160" y="154" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="210" y="154" width="48" height="26" rx="2" fill="#3b82f6"/>
  <rect x="260" y="154" width="82" height="26" rx="2" fill="#f59e0b"/>
  <rect x="344" y="154" width="82" height="26" rx="2" fill="#f59e0b"/>
  <rect x="428" y="154" width="82" height="26" rx="2" fill="#f59e0b"/>
  <rect x="512" y="154" width="82" height="26" rx="2" fill="#f59e0b"/>
  <text x="427" y="172" font-size="11" fill="#ffffff" text-anchor="middle">decode + one quarter of the prompt, four times over</text>
  <rect x="596" y="154" width="48" height="26" rx="2" fill="#3b82f6"/>
  <text x="110" y="200" font-size="11.5" fill="currentColor" opacity="0.8">Total prefill work is the same. No single step is huge, so nobody sees a stall.</text>
  <text x="110" y="222" font-size="11.5" fill="currentColor" opacity="0.8">The cost: the new request waits a little longer for its own first token.</text>
</svg>

Sarathi-Serve ([Agrawal et al., OSDI 2024][sarathi]) is a well-known treatment of this tradeoff. A heavier alternative runs prefill and decode on separate GPUs entirely.

### Reserving memory you do not know you will need

The scheduler has to be sure a request's KV cache will fit before admitting it. But it does not know how long the output will be. The safe move is to reserve enough room for the maximum possible length.

Most requests use far less than the maximum. All that reserved-but-unused memory cannot hold anyone else's KV cache, so the batch stays smaller than the GPU could actually fit — and batch size is where the throughput came from in the first place.

<svg viewBox="0 0 720 270" role="img" aria-label="Reserving KV cache for the maximum output length leaves most of GPU memory reserved but unused, blocking other requests" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="arrD" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="60" y="22" font-size="12.5" font-weight="600" fill="currentColor">Reserve for the worst case</text>
  <rect x="60" y="34" width="150" height="200" rx="4" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <rect x="60" y="34" width="150" height="66" fill="currentColor" opacity="0.08"/>
  <rect x="60" y="82" width="150" height="18" fill="#10b981"/>
  <rect x="60" y="100" width="150" height="66" fill="currentColor" opacity="0.08"/>
  <rect x="60" y="156" width="150" height="10" fill="#10b981"/>
  <rect x="60" y="166" width="150" height="68" fill="currentColor" opacity="0.08"/>
  <rect x="60" y="210" width="150" height="24" fill="#10b981"/>
  <line x1="60" y1="100" x2="210" y2="100" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="60" y1="166" x2="210" y2="166" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <text x="220" y="62" font-size="11" fill="currentColor" opacity="0.75">reserved, never used</text>
  <text x="220" y="94" font-size="11" fill="#10b981">actually used</text>
  <text x="60" y="252" font-size="11.5" fill="#ef4444">3 sequences admitted — a 4th is refused</text>
  <line x1="400" y1="130" x2="450" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrD)"/>
  <text x="470" y="22" font-size="12.5" font-weight="600" fill="currentColor">Hand out memory as it is needed</text>
  <rect x="470" y="34" width="150" height="200" rx="4" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <rect x="470" y="182" width="150" height="18" fill="#10b981"/>
  <rect x="470" y="200" width="150" height="10" fill="#10b981"/>
  <rect x="470" y="210" width="150" height="24" fill="#10b981"/>
  <text x="545" y="112" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">room for more sequences</text>
  <text x="470" y="252" font-size="11.5" fill="#10b981">the same 3 sequences, plus many more</text>
</svg>

This is the problem **PagedAttention** ([Kwon et al., SOSP 2023][paged]) attacks, by managing the KV cache the way an operating system manages virtual memory. That is the next post.

## Summary

- Decode is memory-bound, so batching gets more tokens out of the same weight reads.
- The batch cannot grow forever: compute eventually catches up, and KV cache memory usually runs out first.
- Static batching wastes slots because output lengths vary and the batch waits for its slowest member.
- Continuous batching re-forms the batch every step. That recovers the waste, and introduces prefill stalls and a memory-reservation problem.

## References

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022. [USENIX page][orca]
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv][paged]
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024. [USENIX page][sarathi]

[orca]: https://www.usenix.org/conference/osdi22/presentation/yu
[paged]: https://arxiv.org/abs/2309.06180
[sarathi]: https://www.usenix.org/conference/osdi24/presentation/agrawal
