---
layout: post
title: "PagedAttention: What the KV Cache Learned from Virtual Memory"
date: 2026-11-14 00:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, paged-attention, kv-cache, vllm, memory-management]
description: "Why an LLM server runs out of KV cache long before the arithmetic says it should, and how splitting the cache into fixed-size blocks with a block table recovers the lost memory."
---

Your dashboard says the eight-GPU server has KV capacity for about 120 typical requests. It stops admitting new ones at 80. The model weights did not grow. Nobody deployed anything. The queue builds up while a third of the memory budget sits there, apparently free.

Where did it go?

The [previous post]({% post_url 2026-11-05-llm-inference-batching %}) ended on exactly this problem. Continuous batching gets more tokens out of every byte of weights read, but only if you can keep enough sequences resident, and each sequence needs its KV cache in GPU memory the whole time it runs. The scheduler has to decide whether a new request's cache will fit **before** it admits the request, and it has no idea how long the answer will be.

This post is about the answer that the vLLM team gave in the [PagedAttention paper][paged], which is now how nearly every serving engine manages KV memory. The short version: the memory is not missing, it is *stranded*, and operating systems solved this particular problem decades ago.

## The tempting design

The obvious layout gives each request one contiguous region, sized for the longest answer it is allowed to produce. Addressing is then trivial — base pointer plus position — and the attention kernel reads a flat run of memory.

Since you do not know the output length at admission, "large enough" has to mean the maximum. If your limit is 8,192 tokens, every request reserves 8,192 tokens' worth of space on arrival.

<svg viewBox="0 0 720 290" role="img" aria-label="A contiguous KV arena with three slabs, each reserved for 8K tokens but mostly unused, and a free tail too small for a fourth request" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">One contiguous slab per request, each sized for the worst case</text>
  <rect x="60" y="32" width="12" height="12" rx="2" fill="#10b981"/>
  <text x="78" y="42" font-size="11.5" fill="currentColor">live KV — tokens that exist</text>
  <rect x="300" y="32" width="12" height="12" rx="2" fill="currentColor" opacity="0.14"/>
  <text x="318" y="42" font-size="11.5" fill="currentColor">reserved for tokens that may never exist</text>
  <text x="147" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request A — 8K reserved</text>
  <text x="322" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request B — 8K reserved</text>
  <text x="497" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request C — 8K reserved</text>
  <rect x="60" y="80" width="600" height="60" rx="3" fill="currentColor" opacity="0.08"/>
  <rect x="60" y="80" width="44" height="60" fill="#10b981"/>
  <rect x="235" y="80" width="22" height="60" fill="#10b981"/>
  <rect x="410" y="80" width="153" height="60" fill="#10b981"/>
  <rect x="60" y="80" width="600" height="60" rx="3" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <line x1="235" y1="80" x2="235" y2="140" stroke="currentColor" stroke-width="1.2"/>
  <line x1="410" y1="80" x2="410" y2="140" stroke="currentColor" stroke-width="1.2"/>
  <line x1="585" y1="80" x2="585" y2="140" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="147" y="158" font-size="10.5" fill="currentColor" text-anchor="middle">2K tokens live</text>
  <text x="322" y="158" font-size="10.5" fill="currentColor" text-anchor="middle">1K tokens live</text>
  <text x="497" y="158" font-size="10.5" fill="currentColor" text-anchor="middle">7K tokens live</text>
  <line x1="622" y1="144" x2="622" y2="170" stroke="#ef4444" stroke-width="1.2"/>
  <text x="622" y="186" font-size="10.5" fill="#ef4444" text-anchor="middle">free, and useless</text>
  <text x="0" y="226" font-size="12" fill="currentColor">Live tokens: 10K. Reserved tokens: 24K. Requests admitted: 3.</text>
  <text x="0" y="250" font-size="12" fill="currentColor" opacity="0.8">Request D needs one unbroken slab. The leftover tail is real memory, and it is the wrong shape,</text>
  <text x="0" y="268" font-size="12" fill="currentColor" opacity="0.8">so D waits in the queue while the GPU runs at a third of the concurrency it could.</text>
</svg>

Three separate things are going wrong in that picture, and it is worth naming them, because paging fixes them in different ways:

- **Reserved for later.** Space held for tokens the sequence genuinely will produce. Unavoidable in a contiguous scheme; you cannot hand the space out and take it back.
- **Reserved forever.** Space held for tokens it will never produce, because it stopped at 2,000 and was allowed 8,192. This is the big one.
- **Holes.** When sequences finish at different times, free memory ends up scattered in gaps, each too small to hold a new slab. Total free bytes look fine. None of them are usable.

### How much memory is this, really

Put real numbers on it. [Llama 3.1 70B][llama] in BF16 has 80 layers, 8 KV heads, and head dimension 128, so one cached token costs:

~~~text
2 (K and V) × 80 layers × 8 KV heads × 128 dims × 2 bytes = 327,680 bytes = 320 KiB per token
~~~

A request allowed 8,192 tokens reserves `8,192 × 320 KiB = 2.5 GiB`. If it actually produces 2,048 tokens, it uses `0.625 GiB`. The difference — **1.875 GiB per request** — is reserved and never touched. Across 100 concurrent requests, that is 187.5 GiB of KV capacity held hostage.

That is why the dashboard and the admission controller disagree.

## The idea: stop requiring the memory to be contiguous

The problem is not that the sequence needs a lot of memory. The problem is that it needs memory it does not have *yet*, in one piece, at an address that cannot move.

Operating systems hit this in the 1960s and answered it with virtual memory.

<svg viewBox="0 0 720 250" role="img" aria-label="Virtual memory: a process sees contiguous pages, a page table maps each to a scattered physical frame" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pvm" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">How an operating system solved this</text>
  <text x="100" y="40" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">what the process sees</text>
  <rect x="40" y="48" width="120" height="30" rx="3" fill="#3b82f6"/><text x="100" y="68" font-size="11" fill="#ffffff" text-anchor="middle">page 0</text>
  <rect x="40" y="82" width="120" height="30" rx="3" fill="#6366f1"/><text x="100" y="102" font-size="11" fill="#ffffff" text-anchor="middle">page 1</text>
  <rect x="40" y="116" width="120" height="30" rx="3" fill="#10b981"/><text x="100" y="136" font-size="11" fill="#ffffff" text-anchor="middle">page 2</text>
  <rect x="40" y="150" width="120" height="30" rx="3" fill="#f59e0b"/><text x="100" y="170" font-size="11" fill="#ffffff" text-anchor="middle">page 3</text>
  <line x1="168" y1="114" x2="232" y2="114" stroke="currentColor" stroke-width="1.4" marker-end="url(#pvm)"/>
  <text x="310" y="40" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">page table</text>
  <rect x="240" y="48" width="140" height="132" rx="3" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <text x="310" y="72" font-size="11.5" fill="currentColor" text-anchor="middle">page 0 → frame 7</text>
  <text x="310" y="102" font-size="11.5" fill="currentColor" text-anchor="middle">page 1 → frame 2</text>
  <text x="310" y="132" font-size="11.5" fill="currentColor" text-anchor="middle">page 2 → frame 9</text>
  <text x="310" y="162" font-size="11.5" fill="currentColor" text-anchor="middle">page 3 → frame 4</text>
  <line x1="388" y1="114" x2="432" y2="114" stroke="currentColor" stroke-width="1.4" marker-end="url(#pvm)"/>
  <text x="565" y="40" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">physical memory</text>
  <rect x="440" y="48" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="468" y="67" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">0</text>
  <rect x="500" y="48" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="528" y="67" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">1</text>
  <rect x="560" y="48" width="56" height="30" rx="2" fill="#6366f1"/><text x="588" y="67" font-size="9" fill="#ffffff" text-anchor="middle">2</text>
  <rect x="620" y="48" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="648" y="67" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">3</text>
  <rect x="440" y="82" width="56" height="30" rx="2" fill="#f59e0b"/><text x="468" y="101" font-size="9" fill="#ffffff" text-anchor="middle">4</text>
  <rect x="500" y="82" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="528" y="101" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">5</text>
  <rect x="560" y="82" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="588" y="101" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">6</text>
  <rect x="620" y="82" width="56" height="30" rx="2" fill="#3b82f6"/><text x="648" y="101" font-size="9" fill="#ffffff" text-anchor="middle">7</text>
  <rect x="440" y="116" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="468" y="135" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">8</text>
  <rect x="500" y="116" width="56" height="30" rx="2" fill="#10b981"/><text x="528" y="135" font-size="9" fill="#ffffff" text-anchor="middle">9</text>
  <rect x="560" y="116" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="588" y="135" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">10</text>
  <rect x="620" y="116" width="56" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="648" y="135" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.6">11</text>
  <text x="0" y="212" font-size="12" fill="currentColor">The process sees one unbroken run of pages. The hardware keeps them wherever there was room.</text>
  <text x="0" y="234" font-size="12" fill="currentColor" opacity="0.8">Nothing in the process has to know, and no free frame is ever the wrong shape.</text>
</svg>

Now do the same thing to the KV cache.

Cut each sequence's cache into fixed-size **blocks** — 16 tokens is the usual default. Keep a **block table** per sequence that maps its block 0, 1, 2, … to whichever physical slots in the GPU's KV pool happen to be free. The sequence still has a strict logical order of positions; it just stops caring where those positions live.

<svg viewBox="0 0 720 285" role="img" aria-label="A sequence's logical KV blocks map through a block table to scattered physical slots in the GPU KV pool" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pkv" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The same trick, applied to one sequence's KV cache</text>
  <text x="95" y="42" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">what the sequence sees</text>
  <rect x="25" y="52" width="140" height="30" rx="3" fill="#3b82f6"/><text x="95" y="72" font-size="10.5" fill="#ffffff" text-anchor="middle">block 0 · tokens 0–15</text>
  <rect x="25" y="86" width="140" height="30" rx="3" fill="#6366f1"/><text x="95" y="106" font-size="10.5" fill="#ffffff" text-anchor="middle">block 1 · tokens 16–31</text>
  <rect x="25" y="120" width="140" height="30" rx="3" fill="#10b981"/><text x="95" y="140" font-size="10.5" fill="#ffffff" text-anchor="middle">block 2 · tokens 32–47</text>
  <rect x="25" y="154" width="140" height="30" rx="3" fill="#f59e0b"/><text x="95" y="174" font-size="10.5" fill="#ffffff" text-anchor="middle">block 3 · tokens 48–51</text>
  <line x1="173" y1="118" x2="237" y2="118" stroke="currentColor" stroke-width="1.4" marker-end="url(#pkv)"/>
  <text x="320" y="42" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">block table</text>
  <rect x="245" y="52" width="150" height="132" rx="3" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <text x="320" y="76" font-size="11.5" fill="currentColor" text-anchor="middle">block 0 → slot 7</text>
  <text x="320" y="106" font-size="11.5" fill="currentColor" text-anchor="middle">block 1 → slot 2</text>
  <text x="320" y="136" font-size="11.5" fill="currentColor" text-anchor="middle">block 2 → slot 9</text>
  <text x="320" y="166" font-size="11.5" fill="currentColor" text-anchor="middle">block 3 → slot 4</text>
  <line x1="403" y1="118" x2="447" y2="118" stroke="currentColor" stroke-width="1.4" marker-end="url(#pkv)"/>
  <text x="575" y="42" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">KV pool on the GPU</text>
  <rect x="455" y="52" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="481" y="71" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="511" y="52" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="537" y="71" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="567" y="52" width="52" height="30" rx="2" fill="#6366f1"/><text x="593" y="71" font-size="9" fill="#ffffff" text-anchor="middle">2</text>
  <rect x="623" y="52" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="649" y="71" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="455" y="86" width="52" height="30" rx="2" fill="#f59e0b"/><text x="481" y="105" font-size="9" fill="#ffffff" text-anchor="middle">4</text>
  <rect x="511" y="86" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="537" y="105" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="567" y="86" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="593" y="105" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="623" y="86" width="52" height="30" rx="2" fill="#3b82f6"/><text x="649" y="105" font-size="9" fill="#ffffff" text-anchor="middle">7</text>
  <rect x="455" y="120" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="481" y="139" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="511" y="120" width="52" height="30" rx="2" fill="#10b981"/><text x="537" y="139" font-size="9" fill="#ffffff" text-anchor="middle">9</text>
  <rect x="567" y="120" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="593" y="139" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <rect x="623" y="120" width="52" height="30" rx="2" fill="currentColor" opacity="0.08"/><text x="649" y="139" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.55">free</text>
  <text x="0" y="214" font-size="12" fill="currentColor">The sequence keeps its order. The pool keeps its freedom. The block table is the only thing in between.</text>
  <text x="0" y="236" font-size="12" fill="currentColor" opacity="0.8">To compute attention for token 52, the kernel walks the block table and gathers K and V from slots 7, 2, 9 and 4.</text>
  <text x="0" y="262" font-size="11.5" fill="currentColor" opacity="0.65">16 tokens per block is a common default, not a law. It is a tuning knob — see below.</text>
</svg>

**A precision worth keeping.** Two separate things got invented here, and people routinely conflate them. The **block manager** is an allocator: fixed-size blocks, a free pool, block tables. **PagedAttention** is the attention kernel that can read K and V scattered across those blocks instead of from one flat run. You need both, but they are different pieces, and other runtimes mix and match — [vAttention][vattn] keeps contiguous *virtual* addressing and pages the physical memory underneath, which lets it run unmodified attention kernels.

**Where the analogy breaks.** There is no hardware MMU doing this. The block table is an ordinary tensor the runtime maintains and the kernel indexes, in software, on every attention call. There is no page fault, no demand paging from disk, and no transparent rescue if you get the mapping wrong — a block-table bug is a correctness bug that produces plausible-looking wrong text. And unlike an OS page, a KV block's contents are tied to specific token IDs at specific positions under specific weights, which is what makes the sharing tricks later in this post possible and what makes them dangerous to get wrong.

## Growth, one token at a time

Here is where paging earns its keep during decode. A sequence grows by exactly one token per step. In the contiguous world, that is why you had to reserve for the worst case: growth has nowhere to go. In the paged world, growth touches one block.

<svg viewBox="0 0 720 265" role="img" aria-label="A decoding sequence fills its last block token by token, then takes one new block from the free pool" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pgr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#ef4444"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Growth costs one block, not one slab</text>
  <text x="0" y="55" font-size="11" fill="currentColor">after 50 tokens</text>
  <rect x="150" y="38" width="72" height="24" rx="2" fill="#3b82f6"/><text x="186" y="54" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="230" y="38" width="72" height="24" rx="2" fill="#6366f1"/><text x="266" y="54" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="310" y="38" width="72" height="24" rx="2" fill="#10b981"/><text x="346" y="54" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="390" y="38" width="88" height="24" rx="2" fill="none" stroke="#f59e0b" stroke-width="1.3"/>
  <rect x="392" y="40" width="5" height="20" fill="#f59e0b"/><rect x="397.5" y="40" width="5" height="20" fill="#f59e0b"/>
  <text x="488" y="54" font-size="10.5" fill="currentColor">block 3 holds 2 of its 16 token slots</text>
  <text x="0" y="105" font-size="11" fill="currentColor">14 steps later</text>
  <rect x="150" y="88" width="72" height="24" rx="2" fill="#3b82f6"/><text x="186" y="104" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="230" y="88" width="72" height="24" rx="2" fill="#6366f1"/><text x="266" y="104" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="310" y="88" width="72" height="24" rx="2" fill="#10b981"/><text x="346" y="104" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="390" y="88" width="88" height="24" rx="2" fill="#f59e0b"/><text x="434" y="104" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <text x="488" y="104" font-size="10.5" fill="currentColor">full — the next token has nowhere to go</text>
  <text x="0" y="155" font-size="11" fill="currentColor">next token</text>
  <rect x="150" y="138" width="72" height="24" rx="2" fill="#3b82f6"/><text x="186" y="154" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="230" y="138" width="72" height="24" rx="2" fill="#6366f1"/><text x="266" y="154" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="310" y="138" width="72" height="24" rx="2" fill="#10b981"/><text x="346" y="154" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="390" y="138" width="88" height="24" rx="2" fill="#f59e0b"/><text x="434" y="154" font-size="9" fill="#ffffff" text-anchor="middle">16/16</text>
  <rect x="486" y="138" width="88" height="24" rx="2" fill="none" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="4 3"/>
  <rect x="488" y="140" width="5" height="20" fill="#ef4444"/>
  <text x="582" y="154" font-size="10.5" fill="#ef4444">one new block</text>
  <text x="0" y="200" font-size="11" fill="currentColor">free pool</text>
  <rect x="150" y="186" width="20" height="20" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="174" y="186" width="20" height="20" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="198" y="186" width="20" height="20" rx="2" fill="none" stroke="#ef4444" stroke-width="1.3"/>
  <rect x="222" y="186" width="20" height="20" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="246" y="186" width="20" height="20" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="270" y="186" width="20" height="20" rx="2" fill="currentColor" opacity="0.1"/>
  <line x1="214" y1="182" x2="480" y2="166" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#pgr)"/>
  <text x="302" y="201" font-size="10.5" fill="#ef4444">slot 23, taken on demand</text>
  <text x="0" y="240" font-size="12" fill="currentColor">A sequence only ever holds one partially filled block: its last one. Everything before it is packed solid.</text>
</svg>

That last sentence is the whole result. Tail waste is bounded by one block per sequence, no matter how long the sequence runs or how high the length limit is set.

## What is left of the waste

Keep the Llama 3.1 70B numbers. One 16-token block costs `16 × 320 KiB = 5 MiB` across the model. A sequence that ends mid-block leaves at most 15 unused token slots; averaged over arbitrary ending points, about 7.5, which is **2.34 MiB**.

Compare that with the 1.875 GiB a contiguous slab was holding for the same request.

<svg viewBox="0 0 720 215" role="img" aria-label="Bar chart comparing 1.875 GiB wasted per request under contiguous allocation with 2.34 MiB under 16-token blocks" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Wasted KV memory per request — 8K limit, 2K actual output</text>
  <text x="0" y="78" font-size="11.5" fill="currentColor">contiguous slab</text>
  <rect x="150" y="58" width="440" height="30" rx="2" fill="#ef4444"/>
  <text x="600" y="78" font-size="12" font-weight="600" fill="#ef4444">1.875 GiB</text>
  <text x="0" y="134" font-size="11.5" fill="currentColor">16-token blocks</text>
  <rect x="150" y="114" width="3" height="30" fill="#10b981"/>
  <text x="163" y="134" font-size="12" font-weight="600" fill="#10b981">2.34 MiB — the unused tail of one block</text>
  <text x="0" y="176" font-size="12" fill="currentColor">Across 100 concurrent requests: 187.5 GiB stranded, against roughly 0.23 GiB of block tails.</text>
  <text x="0" y="200" font-size="11.5" fill="currentColor" opacity="0.65">The bars are to scale. The green one is three pixels wide.</text>
</svg>

The external fragmentation disappears too, and for a reason worth stating plainly: **every free block is interchangeable**. A hole can never be the wrong size when all the holes are the same size. The allocator's job collapses from "find a large enough contiguous run" to "pop one off the free list."

So the picture from the top of the post turns into this.

<svg viewBox="0 0 720 230" role="img" aria-label="The same GPU memory holding three sequences under contiguous allocation versus nine under paged allocation" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The same memory, repacked</text>
  <text x="0" y="56" font-size="11.5" fill="currentColor">contiguous</text>
  <text x="660" y="44" font-size="11.5" font-weight="600" fill="currentColor" text-anchor="end">3 sequences</text>
  <rect x="90" y="50" width="570" height="32" rx="3" fill="currentColor" opacity="0.08"/>
  <rect x="90" y="50" width="42" height="32" fill="#10b981"/>
  <rect x="256" y="50" width="21" height="32" fill="#10b981"/>
  <rect x="422" y="50" width="145" height="32" fill="#10b981"/>
  <rect x="90" y="50" width="570" height="32" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <line x1="256" y1="50" x2="256" y2="82" stroke="currentColor" stroke-width="1.1"/>
  <line x1="422" y1="50" x2="422" y2="82" stroke="currentColor" stroke-width="1.1"/>
  <text x="0" y="136" font-size="11.5" fill="currentColor">paged</text>
  <text x="660" y="124" font-size="11.5" font-weight="600" fill="currentColor" text-anchor="end">9 sequences</text>
  <rect x="90" y="130" width="14" height="32" rx="1" fill="#3b82f6"/><rect x="105" y="130" width="14" height="32" rx="1" fill="#3b82f6"/><rect x="120" y="130" width="14" height="32" rx="1" fill="#3b82f6"/><rect x="135" y="130" width="14" height="32" rx="1" fill="#3b82f6"/>
  <rect x="150" y="130" width="14" height="32" rx="1" fill="#6366f1"/><rect x="165" y="130" width="14" height="32" rx="1" fill="#6366f1"/><rect x="180" y="130" width="14" height="32" rx="1" fill="#6366f1"/><rect x="195" y="130" width="14" height="32" rx="1" fill="#6366f1"/><rect x="210" y="130" width="14" height="32" rx="1" fill="#6366f1"/><rect x="225" y="130" width="14" height="32" rx="1" fill="#6366f1"/>
  <rect x="240" y="130" width="14" height="32" rx="1" fill="#10b981"/><rect x="255" y="130" width="14" height="32" rx="1" fill="#10b981"/><rect x="270" y="130" width="14" height="32" rx="1" fill="#10b981"/>
  <rect x="285" y="130" width="14" height="32" rx="1" fill="#f59e0b"/><rect x="300" y="130" width="14" height="32" rx="1" fill="#f59e0b"/><rect x="315" y="130" width="14" height="32" rx="1" fill="#f59e0b"/><rect x="330" y="130" width="14" height="32" rx="1" fill="#f59e0b"/><rect x="345" y="130" width="14" height="32" rx="1" fill="#f59e0b"/>
  <rect x="360" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="375" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="390" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="405" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="420" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="435" y="130" width="14" height="32" rx="1" fill="#ef4444"/><rect x="450" y="130" width="14" height="32" rx="1" fill="#ef4444"/>
  <rect x="465" y="130" width="14" height="32" rx="1" fill="#14b8a6"/><rect x="480" y="130" width="14" height="32" rx="1" fill="#14b8a6"/>
  <rect x="495" y="130" width="14" height="32" rx="1" fill="#8b5cf6"/><rect x="510" y="130" width="14" height="32" rx="1" fill="#8b5cf6"/><rect x="525" y="130" width="14" height="32" rx="1" fill="#8b5cf6"/><rect x="540" y="130" width="14" height="32" rx="1" fill="#8b5cf6"/>
  <rect x="555" y="130" width="14" height="32" rx="1" fill="#ec4899"/><rect x="570" y="130" width="14" height="32" rx="1" fill="#ec4899"/><rect x="585" y="130" width="14" height="32" rx="1" fill="#ec4899"/><rect x="600" y="130" width="14" height="32" rx="1" fill="#ec4899"/><rect x="615" y="130" width="14" height="32" rx="1" fill="#ec4899"/>
  <rect x="630" y="130" width="14" height="32" rx="1" fill="#0ea5e9"/><rect x="645" y="130" width="14" height="32" rx="1" fill="#0ea5e9"/>
  <rect x="90" y="130" width="570" height="32" rx="3" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="0" y="192" font-size="12" fill="currentColor">Nothing about the GPU changed. The allocator stopped holding memory for tokens that do not exist.</text>
  <text x="0" y="214" font-size="11.5" fill="currentColor" opacity="0.65">Illustrative. Your gain depends mostly on how far your length limit sits above your typical output.</text>
</svg>

## Choosing a block size

The one real knob. Blocks are the unit of allocation, so their size trades two costs against each other.

<svg viewBox="0 0 720 235" role="img" aria-label="A 50-token sequence stored in 4-token blocks versus 64-token blocks, showing the tradeoff between table entries and tail waste" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The same 50-token sequence, two block sizes</text>
  <text x="0" y="62" font-size="11.5" fill="currentColor">4 tokens</text>
  <rect x="110" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="136" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="162" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="188" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="214" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="240" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="266" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="292" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="318" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="344" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="370" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="396" y="44" width="24" height="26" rx="2" fill="#10b981"/>
  <rect x="422" y="44" width="12" height="26" rx="1" fill="#10b981"/>
  <rect x="434" y="44" width="12" height="26" rx="1" fill="currentColor" opacity="0.14"/>
  <text x="462" y="62" font-size="11" fill="currentColor">13 block-table entries · 2 token slots wasted</text>
  <text x="0" y="130" font-size="11.5" fill="currentColor">64 tokens</text>
  <rect x="110" y="112" width="262" height="26" rx="2" fill="#10b981"/>
  <rect x="372" y="112" width="74" height="26" rx="2" fill="currentColor" opacity="0.14"/>
  <rect x="110" y="112" width="336" height="26" rx="2" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="462" y="130" font-size="11" fill="currentColor">1 block-table entry · 14 token slots wasted</text>
  <text x="0" y="180" font-size="12" fill="currentColor">Smaller blocks waste less memory and cost more bookkeeping: longer tables, more indexing in the kernel,</text>
  <text x="0" y="198" font-size="12" fill="currentColor">less regular memory access. Larger blocks do the reverse. 16 is the common compromise;</text>
  <text x="0" y="216" font-size="12" fill="currentColor">the right answer depends on your model, your kernels and your length distribution.</text>
</svg>

## The part nobody expected: blocks are shareable

Here is the thing that turned a memory optimization into a feature. Once KV lives in blocks behind a table, **two sequences can point at the same block**.

That matters because identical prefixes are everywhere in real traffic. Ask for five sampled completions of one prompt and all five share the prompt. Run beam search and the beams share everything up to where they diverged. The block manager keeps a reference count per physical block and hands the same slot to all of them.

<svg viewBox="0 0 720 235" role="img" aria-label="Two sampled answers sharing the same physical prompt blocks through their block tables" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="psh" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">One prompt, two answers — the prompt is stored once</text>
  <text x="204" y="50" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">reference count 2</text>
  <rect x="150" y="58" width="104" height="34" rx="3" fill="#3b82f6"/><text x="202" y="79" font-size="10" fill="#ffffff" text-anchor="middle">prompt 0–15</text>
  <rect x="256" y="58" width="104" height="34" rx="3" fill="#3b82f6"/><text x="308" y="79" font-size="10" fill="#ffffff" text-anchor="middle">prompt 16–31</text>
  <line x1="364" y1="70" x2="424" y2="56" stroke="currentColor" stroke-width="1.3" marker-end="url(#psh)"/>
  <line x1="364" y1="82" x2="424" y2="136" stroke="currentColor" stroke-width="1.3" marker-end="url(#psh)"/>
  <text x="430" y="34" font-size="10.5" fill="currentColor" opacity="0.8">sample 1</text>
  <rect x="430" y="42" width="112" height="32" rx="3" fill="#10b981"/><text x="486" y="62" font-size="10" fill="#ffffff" text-anchor="middle">A · 32–47</text>
  <rect x="544" y="42" width="112" height="32" rx="3" fill="#10b981"/><text x="600" y="62" font-size="10" fill="#ffffff" text-anchor="middle">A · 48–63</text>
  <text x="430" y="126" font-size="10.5" fill="currentColor" opacity="0.8">sample 2</text>
  <rect x="430" y="134" width="112" height="32" rx="3" fill="#f59e0b"/><text x="486" y="154" font-size="10" fill="#ffffff" text-anchor="middle">B · 32–47</text>
  <rect x="544" y="134" width="112" height="32" rx="3" fill="#f59e0b"/><text x="600" y="154" font-size="10" fill="#ffffff" text-anchor="middle">B · 48–63</text>
  <text x="0" y="196" font-size="12" fill="currentColor">Two block tables, one copy of the prompt's keys and values. Without blocks, the prompt is stored twice.</text>
  <text x="0" y="218" font-size="12" fill="currentColor" opacity="0.8">With a long shared prompt and several samples, this is most of the request's memory.</text>
</svg>

Sharing works because KV is immutable once written: the keys and values for position 12 never change after position 12 is computed. The only moment that breaks is when two sequences need to write into the *same* partially filled block. The fix is the one the OS uses — copy that block, and only that block.

<svg viewBox="0 0 720 205" role="img" aria-label="Copy-on-write: a half-filled shared block is duplicated when two sequences must both write into it" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pcw" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Copy-on-write, for the one block they both need to change</text>
  <rect x="40" y="62" width="200" height="44" rx="3" fill="#6366f1"/>
  <text x="140" y="80" font-size="10.5" fill="#ffffff" text-anchor="middle">tokens 32–47 · 9 of 16 filled</text>
  <text x="140" y="96" font-size="10.5" fill="#ffffff" text-anchor="middle">reference count 2</text>
  <line x1="248" y1="76" x2="396" y2="58" stroke="currentColor" stroke-width="1.3" marker-end="url(#pcw)"/>
  <line x1="248" y1="92" x2="396" y2="132" stroke="currentColor" stroke-width="1.3" marker-end="url(#pcw)"/>
  <rect x="402" y="38" width="200" height="40" rx="3" fill="#10b981"/>
  <text x="502" y="63" font-size="10.5" fill="#ffffff" text-anchor="middle">copy for A · ref 1</text>
  <rect x="402" y="116" width="200" height="40" rx="3" fill="#f59e0b"/>
  <text x="502" y="141" font-size="10.5" fill="#ffffff" text-anchor="middle">copy for B · ref 1</text>
  <text x="0" y="186" font-size="12" fill="currentColor">One block is duplicated. Every full block before it stays shared, however long the prompt was.</text>
</svg>

The same machinery, pointed at *different requests* rather than different samples of one request, becomes prefix caching: when a thousand users send the same system prompt, the server can keep those blocks and skip the prefill entirely. That deserves its own post, and it is the next one.

## When the pool runs dry anyway

Paging buys a lot of headroom. It does not make memory infinite, and continuous batching will happily admit requests until it is gone. Sequences grow; eventually a running sequence asks for a block and there is none.

<svg viewBox="0 0 720 270" role="img" aria-label="When the free pool is empty the runtime either swaps a victim sequence's blocks to CPU memory or drops them and recomputes later" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="ppr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">When the free pool is empty</text>
  <rect x="20" y="62" width="250" height="60" rx="4" fill="none" stroke="#ef4444" stroke-width="1.4"/>
  <text x="145" y="86" font-size="11.5" fill="#ef4444" text-anchor="middle">a running sequence needs</text>
  <text x="145" y="104" font-size="11.5" fill="#ef4444" text-anchor="middle">one more block, and there is none</text>
  <line x1="278" y1="82" x2="352" y2="70" stroke="currentColor" stroke-width="1.3" marker-end="url(#ppr)"/>
  <line x1="278" y1="102" x2="352" y2="166" stroke="currentColor" stroke-width="1.3" marker-end="url(#ppr)"/>
  <rect x="358" y="42" width="340" height="60" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="372" y="66" font-size="11.5" font-weight="600" fill="#3b82f6">Swap out</text>
  <text x="372" y="86" font-size="11" fill="currentColor">copy a victim's blocks to CPU memory, bring them back later</text>
  <rect x="358" y="138" width="340" height="60" rx="4" fill="none" stroke="#f59e0b" stroke-width="1.4"/>
  <text x="372" y="162" font-size="11.5" font-weight="600" fill="#f59e0b">Recompute</text>
  <text x="372" y="182" font-size="11" fill="currentColor">drop the blocks, redo that prefill when it is rescheduled</text>
  <text x="0" y="232" font-size="12" fill="currentColor">Either way, a sequence is evicted all at once: half its KV on the GPU is no use to anybody.</text>
  <text x="0" y="254" font-size="12" fill="currentColor" opacity="0.8">Preemption count is the metric to watch. A server preempting constantly is admitting work it cannot hold.</text>
</svg>

Swapping trades PCIe bandwidth for the prefill compute; recomputing trades the opposite way. Which is cheaper depends on the sequence length and your interconnect, and good runtimes pick per case.

## What this does and does not buy you

Worth being blunt, because "PagedAttention makes inference faster" is not quite true.

What it directly produces is **more usable KV positions per GB of GPU memory**. Everything else is downstream of that, and only pays out if memory was your binding constraint:

- If you were turning requests away for lack of KV, concurrency rises, the batch gets bigger, and throughput follows — this is where the paper's 2–4× against Orca comes from.
- Time to first token can improve, simply because queued requests get admitted sooner.
- Per-token latency may not improve at all. It can get slightly *worse*: the kernel now does indirection through a block table, and a bigger batch makes each step longer. You traded latency for throughput, which is usually the right trade and is still a trade.
- If your bottleneck is compute, kernel launches, or the network, this changes nothing. A GPU that was already compute-bound stays compute-bound.

Output quality should be identical. Same K and V, same positions, same mask — only the addresses changed. That is also the thing to test hardest: block-table and masking bugs produce fluent, confident, wrong text rather than a crash. Test across block boundaries, cancellations, preemption and resume, and shared-prefix cases specifically.

And do not read "free HBM" as "free blocks". The pool is sized by configuration at startup and shares the GPU with weights, activations and workspace.

## Summary

- A contiguous KV cache forces you to reserve for the longest answer a request is allowed to give, so most of the reservation is never used, and what is left over is scattered into holes.
- Paging cuts the cache into fixed-size blocks and adds a block table, which is exactly what virtual memory does for processes.
- Waste per sequence drops from gigabytes to the tail of one block, and fragmentation disappears because every free block is interchangeable.
- Blocks can be shared with a reference count, which gives parallel sampling, beam search and prefix caching almost for free, with copy-on-write at the one block that diverges.
- When the pool empties, sequences are preempted whole, by swapping to host memory or by recomputing.
- The payoff is concurrency, not raw speed. It shows up as throughput only if KV memory was what you were short of.

Next: if blocks can be shared between two samples of one request, why not between two different requests that start with the same system prompt?

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv][paged]
- [vLLM PagedAttention kernel design][vllmkernel] and the [vLLM architecture walkthrough][vllmarch]
- [NVIDIA TensorRT-LLM KV cache management][trtllm], a different block manager for the same problem
- Prabhu et al., *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*. [arXiv][vattn]
- [Meta Llama 3.1 model configuration][llama], source of the 320 KiB-per-token figure

[paged]: https://arxiv.org/abs/2309.06180
[vllmkernel]: https://docs.vllm.ai/en/latest/design/paged_attention/
[vllmarch]: https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html
[trtllm]: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
[vattn]: https://arxiv.org/abs/2405.04437
[llama]: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
