---
layout: post
title: "Paged attention: why the GPU runs out of KV cache early"
date: 2026-11-14 00:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, paged-attention, kv-cache, vllm, memory-management]
description: "An LLM server stops admitting requests while a third of its KV memory looks free. Where that memory goes, how block tables get it back, and three places where vLLM no longer matches the PagedAttention paper."
---

Your dashboard says the eight-GPU server has KV capacity for about 120 typical requests. It stops admitting new ones at 80. The weights did not grow and nobody deployed anything. The queue builds while a third of the memory budget sits there, apparently free.

The memory is not missing. It is reserved for tokens that do not exist yet, and some of it is reserved for tokens that will never exist.

The [batching article]({% post_url 2026-11-05-llm-inference-batching %}) ended on this problem. Continuous batching pays off only if you keep enough sequences resident, and every resident sequence holds its KV cache on the GPU until it finishes. The scheduler must decide whether a request fits before it admits the request, and it does not know how long the answer will be.

The fix is an allocator, and it borrows its design from virtual memory. The vLLM details here come from the source rather than from the [PagedAttention paper][paged], which is from 2023 and no longer matches the code in three places.

## One slab per request wastes most of the memory

The obvious layout gives each request one contiguous region, sized for the longest answer it may produce. Addressing is then cheap, because the address of position `i` is the base pointer plus `i` times the per-token size.

You do not know the output length when the request arrives. So "large enough" has to mean the maximum. With a limit of 8,192 tokens, every request reserves 8,192 tokens of space on arrival.

<svg viewBox="0 0 720 290" role="img" aria-label="A contiguous KV arena holding three slabs, each reserved for 8K tokens but mostly unused, with a free tail too small for a fourth request" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">One contiguous slab per request, each sized for the worst case</text>
  <rect x="60" y="32" width="12" height="12" rx="2" fill="#10b981"/>
  <text x="78" y="42" font-size="11.5" fill="currentColor">live KV, tokens that exist</text>
  <rect x="300" y="32" width="12" height="12" rx="2" fill="currentColor" opacity="0.14"/>
  <text x="318" y="42" font-size="11.5" fill="currentColor">reserved for tokens that may never exist</text>
  <text x="147" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request A, 8K reserved</text>
  <text x="322" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request B, 8K reserved</text>
  <text x="497" y="72" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">request C, 8K reserved</text>
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
  <text x="622" y="186" font-size="10.5" fill="#ef4444" text-anchor="middle">free and unusable</text>
  <text x="0" y="226" font-size="12" fill="currentColor">Live tokens: 10K. Reserved tokens: 24K. Requests admitted: 3.</text>
  <text x="0" y="250" font-size="12" fill="currentColor" opacity="0.8">Request D needs one unbroken slab. The leftover tail is real memory in the wrong shape,</text>
  <text x="0" y="268" font-size="12" fill="currentColor" opacity="0.8">so D waits while the GPU runs at a third of the concurrency it could.</text>
</svg>

That picture has three problems.

**Space for tokens that arrive later.** A running sequence will use some of its reservation. You cannot lend that space out and take it back, because the region has to stay contiguous.

**Space for tokens that never arrive.** A request allowed 8,192 tokens that stops at 2,000 holds the other 6,192 until it finishes. This is the expensive one.

**Holes.** Sequences finish in a different order than they started, so free memory ends up in gaps. Each gap is too small for a new slab. The total free bytes look fine and none of them are usable.

### Put a number on it

[Llama 3.1 70B][llama] in BF16 has 80 layers, 8 KV heads, and head dimension 128. One cached token costs:

~~~text
2 (K and V) x 80 layers x 8 KV heads x 128 dims x 2 bytes = 327,680 bytes = 320 KiB per token
~~~

A request allowed 8,192 tokens reserves 2.5 GiB. If it produces 2,048 tokens, it uses 0.625 GiB. The other 1.875 GiB stays reserved and untouched. Across 100 concurrent requests that is 187.5 GiB of KV capacity that nobody can use.

## Virtual memory already solved this

The sequence does not need a lot of memory. It needs memory it does not have yet, in one piece, at an address that cannot move. Operating systems had the same problem in the 1960s.

<svg viewBox="0 0 720 250" role="img" aria-label="Virtual memory: a process sees contiguous pages, a page table maps each page to a scattered physical frame" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pvm" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">What an operating system does</text>
  <text x="100" y="40" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">what the process sees</text>
  <rect x="40" y="48" width="120" height="30" rx="3" fill="#3b82f6"/><text x="100" y="68" font-size="11" fill="#ffffff" text-anchor="middle">page 0</text>
  <rect x="40" y="82" width="120" height="30" rx="3" fill="#6366f1"/><text x="100" y="102" font-size="11" fill="#ffffff" text-anchor="middle">page 1</text>
  <rect x="40" y="116" width="120" height="30" rx="3" fill="#10b981"/><text x="100" y="136" font-size="11" fill="#ffffff" text-anchor="middle">page 2</text>
  <rect x="40" y="150" width="120" height="30" rx="3" fill="#f59e0b"/><text x="100" y="170" font-size="11" fill="#ffffff" text-anchor="middle">page 3</text>
  <line x1="168" y1="114" x2="232" y2="114" stroke="currentColor" stroke-width="1.4" marker-end="url(#pvm)"/>
  <text x="310" y="40" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">page table</text>
  <rect x="240" y="48" width="140" height="132" rx="3" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <text x="310" y="72" font-size="11.5" fill="currentColor" text-anchor="middle">page 0 to frame 7</text>
  <text x="310" y="102" font-size="11.5" fill="currentColor" text-anchor="middle">page 1 to frame 2</text>
  <text x="310" y="132" font-size="11.5" fill="currentColor" text-anchor="middle">page 2 to frame 9</text>
  <text x="310" y="162" font-size="11.5" fill="currentColor" text-anchor="middle">page 3 to frame 4</text>
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
  <text x="0" y="234" font-size="12" fill="currentColor" opacity="0.8">No free frame is ever the wrong shape, because every frame is the same size.</text>
</svg>

Now do the same thing to the KV cache. Cut each sequence's cache into fixed-size blocks. Keep a block table per sequence that maps its block 0, 1, 2 and so on to whatever physical slots are free. The sequence keeps a strict order of positions, but those positions no longer have to sit next to each other in memory.

<svg viewBox="0 0 720 300" role="img" aria-label="A sequence's logical KV blocks map through a block table to scattered slots in the GPU KV pool, with the lookup animating one block at a time" style="width:100%;height:auto;max-width:720px;">
  <style>
    @keyframes pa-g0 { 0%, 20% { opacity: 1 } 24%, 100% { opacity: 0 } }
    @keyframes pa-g1 { 0%, 20% { opacity: 0 } 24%, 45% { opacity: 1 } 49%, 100% { opacity: 0 } }
    @keyframes pa-g2 { 0%, 45% { opacity: 0 } 49%, 70% { opacity: 1 } 74%, 100% { opacity: 0 } }
    @keyframes pa-g3 { 0%, 70% { opacity: 0 } 74%, 95% { opacity: 1 } 99%, 100% { opacity: 0 } }
    .pa-gather { opacity: 0 }
    .pa-g0 { animation: pa-g0 5s linear infinite }
    .pa-g1 { animation: pa-g1 5s linear infinite }
    .pa-g2 { animation: pa-g2 5s linear infinite }
    .pa-g3 { animation: pa-g3 5s linear infinite }
    @media (prefers-reduced-motion: reduce) {
      .pa-gather { animation: none !important; opacity: 1 }
    }
  </style>
  <defs><marker id="pkv" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The same idea, applied to one sequence's KV cache</text>
  <text x="95" y="42" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">what the sequence sees</text>
  <rect x="25" y="52" width="140" height="30" rx="3" fill="#3b82f6"/><text x="95" y="72" font-size="10.5" fill="#ffffff" text-anchor="middle">block 0, tokens 0-63</text>
  <rect x="25" y="86" width="140" height="30" rx="3" fill="#6366f1"/><text x="95" y="106" font-size="10.5" fill="#ffffff" text-anchor="middle">block 1, tokens 64-127</text>
  <rect x="25" y="120" width="140" height="30" rx="3" fill="#10b981"/><text x="95" y="140" font-size="10.5" fill="#ffffff" text-anchor="middle">block 2, tokens 128-191</text>
  <rect x="25" y="154" width="140" height="30" rx="3" fill="#f59e0b"/><text x="95" y="174" font-size="10.5" fill="#ffffff" text-anchor="middle">block 3, filling</text>
  <line x1="173" y1="118" x2="237" y2="118" stroke="currentColor" stroke-width="1.4" marker-end="url(#pkv)"/>
  <text x="320" y="42" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.8">block table</text>
  <rect x="245" y="52" width="150" height="132" rx="3" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <rect class="pa-gather pa-g0" x="249" y="58" width="142" height="24" rx="2" fill="#3b82f6" opacity="0"/>
  <rect class="pa-gather pa-g1" x="249" y="88" width="142" height="24" rx="2" fill="#6366f1" opacity="0"/>
  <rect class="pa-gather pa-g2" x="249" y="118" width="142" height="24" rx="2" fill="#10b981" opacity="0"/>
  <rect class="pa-gather pa-g3" x="249" y="148" width="142" height="24" rx="2" fill="#f59e0b" opacity="0"/>
  <text x="320" y="76" font-size="11.5" fill="currentColor" text-anchor="middle">block 0 to slot 7</text>
  <text x="320" y="106" font-size="11.5" fill="currentColor" text-anchor="middle">block 1 to slot 2</text>
  <text x="320" y="136" font-size="11.5" fill="currentColor" text-anchor="middle">block 2 to slot 9</text>
  <text x="320" y="166" font-size="11.5" fill="currentColor" text-anchor="middle">block 3 to slot 4</text>
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
  <rect class="pa-gather pa-g0" x="621" y="84" width="56" height="34" rx="3" fill="none" stroke="#3b82f6" stroke-width="3" opacity="0"/>
  <rect class="pa-gather pa-g1" x="565" y="50" width="56" height="34" rx="3" fill="none" stroke="#6366f1" stroke-width="3" opacity="0"/>
  <rect class="pa-gather pa-g2" x="509" y="118" width="56" height="34" rx="3" fill="none" stroke="#10b981" stroke-width="3" opacity="0"/>
  <rect class="pa-gather pa-g3" x="453" y="84" width="56" height="34" rx="3" fill="none" stroke="#f59e0b" stroke-width="3" opacity="0"/>
  <text x="0" y="222" font-size="12" fill="currentColor">To compute attention for the next token, the kernel walks the block table and reads K and V from each slot.</text>
  <text x="0" y="244" font-size="12" fill="currentColor" opacity="0.8">The animation steps through that walk. The order of the slots in the pool does not matter.</text>
  <text x="0" y="272" font-size="11.5" fill="currentColor" opacity="0.65">Block sizes here are 64 tokens, which is vLLM's current default.</text>
</svg>

Two different things came out of that paper. The **block manager** is an allocator. It owns fixed-size blocks, a free list, and the per-sequence block tables. **PagedAttention** is the attention kernel that reads K and V scattered across those blocks instead of from one flat run. You need both, and they are different code.

That second piece has aged. On NVIDIA hardware vLLM now runs FlashAttention by default, and the backend hands it a `block_table` tensor directly. The hand-written `paged_attention` CUDA kernel from the paper still exists, mostly on ROCm and older paths.

The analogy stops at the hardware. No MMU is involved. The block table is an ordinary tensor that the runtime maintains and the kernel indexes on every attention call, in software. There is no page fault and no demand paging from disk. A wrong mapping does not trap either, so the model reads the wrong keys and values and generates text that reads perfectly well.

## Growth costs one block

A sequence grows by one token per step. The contiguous layout had to reserve for the worst case because a region cannot grow into memory that another request owns. With blocks, growth touches one block.

<svg viewBox="0 0 720 230" role="img" aria-label="Animation of a sequence filling its last KV block token by token, then taking a new block from the free pool" style="width:100%;height:auto;max-width:720px;">
  <style>
    @keyframes pa-grow { 0% { transform: scaleX(0) } 55%, 100% { transform: scaleX(1) } }
    @keyframes pa-grow2 { 0%, 62% { transform: scaleX(0) } 70%, 100% { transform: scaleX(0.125) } }
    @keyframes pa-appear { 0%, 60% { opacity: 0 } 68%, 100% { opacity: 1 } }
    @keyframes pa-taken { 0%, 60% { opacity: 0.1 } 68%, 100% { opacity: 1 } }
    .pa-bar { transform-box: fill-box; transform-origin: left center }
    .pa-b1 { animation: pa-grow 7s steps(16, end) infinite }
    .pa-b2 { animation: pa-grow2 7s steps(8, end) infinite }
    .pa-new { animation: pa-appear 7s linear infinite }
    .pa-slot { animation: pa-taken 7s linear infinite }
    @media (prefers-reduced-motion: reduce) {
      .pa-bar, .pa-new, .pa-slot { animation: none !important }
      .pa-b2 { transform: scaleX(0.125) }
    }
  </style>
  <defs><marker id="pgr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#ef4444"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Growth takes one block, not one slab</text>
  <text x="0" y="70" font-size="11" fill="currentColor">the sequence</text>
  <rect x="120" y="52" width="88" height="30" rx="2" fill="#3b82f6"/><text x="164" y="71" font-size="9" fill="#ffffff" text-anchor="middle">full</text>
  <rect x="212" y="52" width="88" height="30" rx="2" fill="#6366f1"/><text x="256" y="71" font-size="9" fill="#ffffff" text-anchor="middle">full</text>
  <rect x="304" y="52" width="88" height="30" rx="2" fill="#10b981"/><text x="348" y="71" font-size="9" fill="#ffffff" text-anchor="middle">full</text>
  <rect x="398" y="52" width="88" height="30" rx="2" fill="currentColor" opacity="0.1"/>
  <rect class="pa-bar pa-b1" x="398" y="52" width="88" height="30" rx="2" fill="#f59e0b"/>
  <rect x="398" y="52" width="88" height="30" rx="2" fill="none" stroke="#f59e0b" stroke-width="1.3"/>
  <g class="pa-new">
    <rect x="492" y="52" width="88" height="30" rx="2" fill="currentColor" opacity="0.1"/>
    <rect class="pa-bar pa-b2" x="492" y="52" width="88" height="30" rx="2" fill="#ef4444"/>
    <rect x="492" y="52" width="88" height="30" rx="2" fill="none" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="4 3"/>
    <text x="592" y="71" font-size="10.5" fill="#ef4444">new block</text>
  </g>
  <text x="0" y="128" font-size="11" fill="currentColor">free pool</text>
  <rect x="120" y="114" width="22" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="146" y="114" width="22" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <rect class="pa-slot" x="172" y="114" width="22" height="22" rx="2" fill="#ef4444" opacity="0.1"/>
  <rect x="198" y="114" width="22" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="224" y="114" width="22" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="250" y="114" width="22" height="22" rx="2" fill="currentColor" opacity="0.1"/>
  <line x1="194" y1="112" x2="486" y2="88" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#pgr)"/>
  <text x="290" y="130" font-size="10.5" fill="#ef4444">one slot, taken when the last block fills</text>
  <text x="0" y="178" font-size="12" fill="currentColor">A sequence holds at most one partly filled block, which is always its last one.</text>
  <text x="0" y="200" font-size="12" fill="currentColor" opacity="0.8">Every block before it is packed solid. The length limit no longer costs anything up front.</text>
</svg>

Wasted space per sequence is therefore less than one block, no matter how long the sequence runs or how high you set the length limit.

## What waste is left

vLLM's default block size is 64 tokens, set by `DEFAULT_BLOCK_SIZE` in `vllm/config/cache.py`. For the Llama 3.1 70B numbers above, one block costs 64 x 320 KiB = 20 MiB across the model. A sequence that stops partway through its last block leaves at most 63 unused token slots. Averaged over arbitrary stopping points that is about 32 slots, or 10 MiB.

Compare that with the 1.875 GiB the contiguous slab was holding for the same request.

<svg viewBox="0 0 720 215" role="img" aria-label="Bar chart comparing 1.875 GiB wasted per request under contiguous allocation with about 10 MiB under 64-token blocks" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Wasted KV memory per request, 8K limit and 2K actual output</text>
  <text x="0" y="78" font-size="11.5" fill="currentColor">contiguous slab</text>
  <rect x="150" y="58" width="440" height="30" rx="2" fill="#ef4444"/>
  <text x="600" y="78" font-size="12" font-weight="600" fill="#ef4444">1.875 GiB</text>
  <text x="0" y="134" font-size="11.5" fill="currentColor">64-token blocks</text>
  <rect x="150" y="114" width="2" height="30" fill="#10b981"/>
  <text x="162" y="134" font-size="12" font-weight="600" fill="#10b981">about 10 MiB, the unused tail of one block</text>
  <text x="0" y="176" font-size="12" fill="currentColor">Across 100 concurrent requests: 187.5 GiB stranded against roughly 1 GiB of block tails.</text>
  <text x="0" y="200" font-size="11.5" fill="currentColor" opacity="0.65">The bars are to scale. The green one is two pixels wide.</text>
</svg>

The holes disappear too. Every free block is interchangeable, so a hole can never be the wrong size. The allocator's job drops from "find a large enough contiguous run" to "take the next block off the free list".

<svg viewBox="0 0 720 230" role="img" aria-label="The same GPU memory holding three sequences under contiguous allocation and nine under paged allocation" style="width:100%;height:auto;max-width:720px;">
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
  <text x="0" y="214" font-size="11.5" fill="currentColor" opacity="0.65">Illustrative. Your gain depends on how far your length limit sits above your typical output.</text>
</svg>

## Picking a block size

Blocks are the unit of allocation, so their size trades two costs against each other.

<svg viewBox="0 0 720 235" role="img" aria-label="The same 50-token sequence stored in 4-token blocks and in 64-token blocks, showing the tradeoff between table entries and tail waste" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The same 50-token sequence at two block sizes</text>
  <text x="0" y="62" font-size="11.5" fill="currentColor">4 tokens</text>
  <rect x="110" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="136" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="162" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="188" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="214" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="240" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="266" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="292" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="318" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="344" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="370" y="44" width="24" height="26" rx="2" fill="#10b981"/><rect x="396" y="44" width="24" height="26" rx="2" fill="#10b981"/>
  <rect x="422" y="44" width="12" height="26" rx="1" fill="#10b981"/>
  <rect x="434" y="44" width="12" height="26" rx="1" fill="currentColor" opacity="0.14"/>
  <text x="462" y="62" font-size="11" fill="currentColor">13 block-table entries, 2 token slots wasted</text>
  <text x="0" y="130" font-size="11.5" fill="currentColor">64 tokens</text>
  <rect x="110" y="112" width="262" height="26" rx="2" fill="#10b981"/>
  <rect x="372" y="112" width="74" height="26" rx="2" fill="currentColor" opacity="0.14"/>
  <rect x="110" y="112" width="336" height="26" rx="2" fill="none" stroke="currentColor" stroke-width="1.2"/>
  <text x="462" y="130" font-size="11" fill="currentColor">1 block-table entry, 14 token slots wasted</text>
  <text x="0" y="180" font-size="12" fill="currentColor">Small blocks waste less memory and cost more bookkeeping. The table gets longer, the kernel does more</text>
  <text x="0" y="198" font-size="12" fill="currentColor">indexing, and memory access gets less regular. Large blocks trade the other way.</text>
  <text x="0" y="216" font-size="12" fill="currentColor" opacity="0.8">vLLM moved its default from 16 to 64 as attention kernels got better at large blocks.</text>
</svg>

Block size also sets the granularity of cache hits. Two requests can share a block only if they agree on all 64 of its tokens.

## Sharing blocks between requests

Once KV lives in blocks behind a table, two sequences can point at the same block. vLLM tracks this with `ref_cnt` on `KVCacheBlock` in `vllm/v1/core/kv_cache_utils.py`. A block with `ref_cnt` above zero cannot be reused by anyone else.

Identical prefixes are common in real traffic. Every request from one application starts with the same system prompt. Every turn of a chat repeats the whole conversation so far.

<svg viewBox="0 0 720 235" role="img" aria-label="Two requests with the same system prompt pointing at the same physical blocks through their own block tables" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="psh" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Two requests, one copy of the shared prefix</text>
  <text x="204" y="50" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">ref_cnt = 2</text>
  <rect x="150" y="58" width="104" height="34" rx="3" fill="#3b82f6"/><text x="202" y="79" font-size="10" fill="#ffffff" text-anchor="middle">system prompt</text>
  <rect x="256" y="58" width="104" height="34" rx="3" fill="#3b82f6"/><text x="308" y="79" font-size="10" fill="#ffffff" text-anchor="middle">system prompt</text>
  <line x1="364" y1="70" x2="424" y2="56" stroke="currentColor" stroke-width="1.3" marker-end="url(#psh)"/>
  <line x1="364" y1="82" x2="424" y2="136" stroke="currentColor" stroke-width="1.3" marker-end="url(#psh)"/>
  <text x="430" y="34" font-size="10.5" fill="currentColor" opacity="0.8">request 1</text>
  <rect x="430" y="42" width="112" height="32" rx="3" fill="#10b981"/><text x="486" y="62" font-size="10" fill="#ffffff" text-anchor="middle">its own tokens</text>
  <rect x="544" y="42" width="112" height="32" rx="3" fill="#10b981"/><text x="600" y="62" font-size="10" fill="#ffffff" text-anchor="middle">its own tokens</text>
  <text x="430" y="126" font-size="10.5" fill="currentColor" opacity="0.8">request 2</text>
  <rect x="430" y="134" width="112" height="32" rx="3" fill="#f59e0b"/><text x="486" y="154" font-size="10" fill="#ffffff" text-anchor="middle">its own tokens</text>
  <rect x="544" y="134" width="112" height="32" rx="3" fill="#f59e0b"/><text x="600" y="154" font-size="10" fill="#ffffff" text-anchor="middle">its own tokens</text>
  <text x="0" y="196" font-size="12" fill="currentColor">Two block tables, one copy of the prefix keys and values. Request 2 also skips the prefill for those tokens.</text>
  <text x="0" y="218" font-size="12" fill="currentColor" opacity="0.8">vLLM turns this on by default. enable_prefix_caching is True in vllm/config/cache.py.</text>
</svg>

Sharing works because KV is immutable once written. The keys and values for position 12 never change after the model computes position 12. vLLM finds shareable blocks by hashing the token IDs a block covers and looking the hash up in `cached_block_hash_to_block`.

When a request finishes, its blocks go back on a free list that keeps their contents and their hash. `FreeKVCacheBlockQueue` orders that list least-recently-used first, so a freed block survives until someone needs the space. A later request with the same prefix can still hit it.

## When the pool is empty

Paging gives you more headroom, not infinite memory. Continuous batching admits requests until that headroom is gone, and the sequences already running keep growing. Eventually one of them asks for a block and the free list is empty.

<svg viewBox="0 0 720 235" role="img" aria-label="When the free pool is empty vLLM frees a victim request's blocks, resets its progress, and puts it back on the waiting queue" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="ppr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">What vLLM does when the free list is empty</text>
  <rect x="10" y="54" width="180" height="56" rx="4" fill="none" stroke="#ef4444" stroke-width="1.4"/>
  <text x="100" y="78" font-size="11" fill="#ef4444" text-anchor="middle">a running sequence needs</text>
  <text x="100" y="94" font-size="11" fill="#ef4444" text-anchor="middle">a block, and there is none</text>
  <line x1="196" y1="82" x2="232" y2="82" stroke="currentColor" stroke-width="1.3" marker-end="url(#ppr)"/>
  <rect x="238" y="54" width="180" height="56" rx="4" fill="none" stroke="#f59e0b" stroke-width="1.4"/>
  <text x="328" y="78" font-size="11" fill="currentColor" text-anchor="middle">pick a victim: the last</text>
  <text x="328" y="94" font-size="11" fill="currentColor" text-anchor="middle">request in the running list</text>
  <line x1="424" y1="82" x2="460" y2="82" stroke="currentColor" stroke-width="1.3" marker-end="url(#ppr)"/>
  <rect x="466" y="54" width="244" height="56" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.4"/>
  <text x="588" y="72" font-size="11" fill="currentColor" text-anchor="middle">free all of its blocks, set</text>
  <text x="588" y="88" font-size="11" fill="currentColor" text-anchor="middle">num_computed_tokens to 0, and</text>
  <text x="588" y="104" font-size="11" fill="currentColor" text-anchor="middle">put it back on the waiting queue</text>
  <text x="0" y="152" font-size="12" fill="currentColor">The victim loses all of its KV at once. Half a sequence of KV on the GPU is no use to anyone.</text>
  <text x="0" y="174" font-size="12" fill="currentColor">When it runs again it redoes its prefill, though prefix caching often returns some of its old blocks.</text>
  <text x="0" y="196" font-size="12" fill="currentColor" opacity="0.8">Watch the preemption counter. A server that preempts often is admitting work it cannot hold.</text>
  <text x="0" y="222" font-size="11.5" fill="currentColor" opacity="0.65">Source: _preempt_request in vllm/v1/core/sched/scheduler.py</text>
</svg>

The paper also describes swapping the victim's blocks to CPU memory and copying them back later. The V1 engine dropped that path. Recompute is the only one left. The victim's old blocks are often still on the free list when it returns, so the recompute is usually cheaper than a cold prefill.

## Where the paper and the current code differ

The paper is from 2023. Three of its mechanisms are no longer what the code does.

**Block size.** The paper and most write-ups use 16 tokens. `DEFAULT_BLOCK_SIZE` in `vllm/config/cache.py` is now 64. Larger blocks suit the attention kernels that shipped since.

**Preemption.** The paper offers swap-out and recompute. V1 implements recompute only. `_preempt_request` frees the blocks, sets `num_computed_tokens` to 0, and re-queues the request.

**Sharing.** The paper forks a sequence and copies the one block both children need to write, which is classic copy-on-write. V1 has no fork path. A request with `n` above 1 becomes `n` independent child requests in `vllm/v1/engine/parallel_sampling.py`, and they share the prompt through the ordinary prefix cache. Beam search moved out of the engine entirely and now sits in the entrypoints layer. Block copies still exist in V1, but only for a narrow case, which is a partial prefix-cache hit that lands inside a block.

In all three cases the blocks and the block table stayed, and the mechanism built on top of them was replaced by hashing and a shared cache.

## What paging changes

"PagedAttention makes inference faster" is not quite right. What it produces is more usable KV positions per GB of GPU memory. Everything else follows from that, and only if memory was the limit.

If you were turning requests away for lack of KV, more requests now fit, so the batch gets bigger and throughput rises. That is where the paper's 2x to 4x against Orca comes from. Time to first token can improve too, because queued requests get admitted sooner.

Per-token latency may not improve at all. It can get slightly worse, since the kernel now indexes through a block table and a larger batch makes each step longer. That is a trade of latency for throughput, and usually the right one.

If your bottleneck is compute, kernel launches, or the network, none of this helps. A GPU that was compute-bound stays compute-bound.

Output quality should not change. Same K and V, same positions, same mask, different addresses. Test that hardest, because a block-table or masking bug produces wrong text instead of a crash. Test across block boundaries, cancellations, preemption and resume, and shared prefixes.

Free HBM is not free blocks. vLLM sizes the pool once at startup from `gpu_memory_utilization`, and the pool shares the GPU with weights, activations, and workspace.

## Summary

- A contiguous KV cache forces you to reserve for the longest answer a request may give. Most of that reservation goes unused, and what is left over scatters into holes.
- Paging cuts the cache into fixed-size blocks and adds a block table, which is what virtual memory does for processes.
- Waste per sequence drops to the tail of one block. Holes disappear, because every free block is interchangeable.
- Blocks are shared by reference count and found by hashing their tokens. vLLM enables prefix caching by default.
- When the pool empties, vLLM frees a whole sequence's blocks and recomputes it later. The swap path from the paper is gone.
- The gain is concurrency. It shows up as throughput only if KV memory was the limit.

Next in this series: prefix caching, and what it costs to keep blocks around after a request finishes.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv][paged]
- vLLM source, in `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`, and `vllm/v1/core/sched/scheduler.py`.
- [vLLM PagedAttention kernel design][vllmkernel] and the [vLLM architecture walkthrough][vllmarch]
- [NVIDIA TensorRT-LLM KV cache management][trtllm], a different block manager for the same problem
- Prabhu et al., *vAttention*, which keeps contiguous virtual addressing and pages the physical memory underneath. [arXiv][vattn]
- [Meta Llama 3.1 model configuration][llama], the source of the 320 KiB per token figure

[paged]: https://arxiv.org/abs/2309.06180
[vllmkernel]: https://docs.vllm.ai/en/latest/design/paged_attention/
[vllmarch]: https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html
[trtllm]: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
[vattn]: https://arxiv.org/abs/2405.04437
[llama]: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
