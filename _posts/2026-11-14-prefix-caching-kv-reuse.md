---
layout: post
title: "Prefix caching: skipping prefill you have already done"
date: 2026-11-14 12:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, prefix-caching, kv-cache, vllm, ttft]
description: "Every request sends the same 4,000-token system prompt, and the GPU prefills it every time. How block hashing turns the KV free list into a cache, what it will not match, and why one salted block isolates a whole tenant."
---

An internal API sends the same 4,096-token policy document before every question. One user asks for a summary. The next asks about a single clause. The GPU processes that identical document twice, and both users wait while it does.

Nothing about the second prefill produces new information. The keys and values for those 4,096 positions are the same both times. The tokens are the same, and attention is causal, so a position depends only on the tokens before it.

The [paged attention article]({% post_url 2026-11-14-paged-attention-kv-memory %}) left one detail hanging. When a request finishes, its blocks go back on a free list that keeps their contents and their hash. That detail is the whole feature. The vLLM details here come from the source rather than from the docs, which lag the code in a few places.

## The work being repeated

Prefill processes the prompt and builds its KV state. Decode then generates tokens one at a time. Two requests that start with the same tokens do the same prefill work for the shared part.

<svg viewBox="0 0 720 250" role="img" aria-label="Two requests with the same 4096-token document each run a full prefill, then the second one reuses the first one's blocks and prefills only its question" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Same document, two questions</text>
  <text x="0" y="52" font-size="11" fill="currentColor" opacity="0.8">without reuse</text>
  <text x="0" y="82" font-size="11.5" fill="currentColor">request A</text>
  <rect x="110" y="66" width="400" height="24" rx="2" fill="#3b82f6"/><text x="310" y="83" font-size="10.5" fill="#ffffff" text-anchor="middle">prefill 4,096 document tokens</text>
  <rect x="512" y="66" width="52" height="24" rx="2" fill="#10b981"/><text x="538" y="83" font-size="9.5" fill="#ffffff" text-anchor="middle">question</text>
  <text x="0" y="116" font-size="11.5" fill="currentColor">request B</text>
  <rect x="110" y="100" width="400" height="24" rx="2" fill="#3b82f6"/><text x="310" y="117" font-size="10.5" fill="#ffffff" text-anchor="middle">prefill the same 4,096 tokens again</text>
  <rect x="512" y="100" width="52" height="24" rx="2" fill="#f59e0b"/><text x="538" y="117" font-size="9.5" fill="#ffffff" text-anchor="middle">question</text>
  <text x="0" y="162" font-size="11" fill="currentColor" opacity="0.8">with reuse</text>
  <text x="0" y="192" font-size="11.5" fill="currentColor">request A</text>
  <rect x="110" y="176" width="400" height="24" rx="2" fill="#3b82f6"/><text x="310" y="193" font-size="10.5" fill="#ffffff" text-anchor="middle">prefill 4,096 document tokens</text>
  <rect x="512" y="176" width="52" height="24" rx="2" fill="#10b981"/><text x="538" y="193" font-size="9.5" fill="#ffffff" text-anchor="middle">question</text>
  <text x="0" y="226" font-size="11.5" fill="currentColor">request B</text>
  <rect x="110" y="210" width="400" height="24" rx="2" fill="currentColor" opacity="0.1"/>
  <rect x="110" y="210" width="400" height="24" rx="2" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="310" y="227" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.75">read A's blocks, compute nothing</text>
  <rect x="512" y="210" width="52" height="24" rx="2" fill="#f59e0b"/><text x="538" y="227" font-size="9.5" fill="#ffffff" text-anchor="middle">question</text>
  <text x="580" y="227" font-size="10.5" fill="#10b981">B starts here</text>
</svg>

Caching the answer instead would not help. The users ask different questions, so the answers differ. What repeats is the prompt, and what the GPU can keep is the KV state for the prompt.

## What counts as the same prefix

This is an exact match on token IDs, in order, from position zero. Similar wording does not count. "The policy says" and "This policy says" tokenize differently at the first word, so the match ends there.

The match also ends for good at the first difference. If two prompts agree for 1,000 tokens, diverge at token 1,001, then contain the same paragraph later, that later paragraph is not reusable. Its keys and values were computed after different earlier tokens, so they hold different numbers.

<svg viewBox="0 0 720 230" role="img" aria-label="Animation scanning two prompts block by block, matching until the first difference, after which nothing matches" style="width:100%;height:auto;max-width:720px;">
  <style>
    @keyframes pc-m1 { 0%, 7% { opacity: 0 } 11%, 90% { opacity: 1 } 95%, 100% { opacity: 0 } }
    @keyframes pc-m2 { 0%, 19% { opacity: 0 } 23%, 90% { opacity: 1 } 95%, 100% { opacity: 0 } }
    @keyframes pc-m3 { 0%, 31% { opacity: 0 } 35%, 90% { opacity: 1 } 95%, 100% { opacity: 0 } }
    @keyframes pc-m4 { 0%, 43% { opacity: 0 } 47%, 90% { opacity: 1 } 95%, 100% { opacity: 0 } }
    @keyframes pc-x  { 0%, 55% { opacity: 0 } 59%, 90% { opacity: 1 } 95%, 100% { opacity: 0 } }
    .pc-s { opacity: 0 }
    .pc-s1 { animation: pc-m1 6s linear infinite }
    .pc-s2 { animation: pc-m2 6s linear infinite }
    .pc-s3 { animation: pc-m3 6s linear infinite }
    .pc-s4 { animation: pc-m4 6s linear infinite }
    .pc-sx { animation: pc-x 6s linear infinite }
    @media (prefers-reduced-motion: reduce) { .pc-s { animation: none !important; opacity: 1 } }
  </style>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">The match ends at the first difference and does not resume</text>
  <text x="0" y="62" font-size="11.5" fill="currentColor">prompt A</text>
  <rect x="96" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="142" y="61" font-size="10" fill="currentColor" text-anchor="middle">system</text>
  <rect x="192" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="238" y="61" font-size="10" fill="currentColor" text-anchor="middle">doc 1</text>
  <rect x="288" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="334" y="61" font-size="10" fill="currentColor" text-anchor="middle">doc 2</text>
  <rect x="384" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="430" y="61" font-size="10" fill="currentColor" text-anchor="middle">doc 3</text>
  <rect x="480" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="526" y="61" font-size="10" fill="currentColor" text-anchor="middle">question A</text>
  <rect x="576" y="44" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="622" y="61" font-size="10" fill="currentColor" text-anchor="middle">same tail</text>
  <text x="0" y="112" font-size="11.5" fill="currentColor">prompt B</text>
  <rect x="96" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="142" y="111" font-size="10" fill="currentColor" text-anchor="middle">system</text>
  <rect x="192" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="238" y="111" font-size="10" fill="currentColor" text-anchor="middle">doc 1</text>
  <rect x="288" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="334" y="111" font-size="10" fill="currentColor" text-anchor="middle">doc 2</text>
  <rect x="384" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="430" y="111" font-size="10" fill="currentColor" text-anchor="middle">doc 3</text>
  <rect x="480" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="526" y="111" font-size="10" fill="currentColor" text-anchor="middle">question B</text>
  <rect x="576" y="94" width="92" height="26" rx="2" fill="currentColor" opacity="0.12"/><text x="622" y="111" font-size="10" fill="currentColor" text-anchor="middle">same tail</text>
  <g class="pc-s pc-s1"><rect x="96" y="44" width="92" height="76" rx="3" fill="#10b981" opacity="0.22"/><rect x="96" y="44" width="92" height="76" rx="3" fill="none" stroke="#10b981" stroke-width="2"/></g>
  <g class="pc-s pc-s2"><rect x="192" y="44" width="92" height="76" rx="3" fill="#10b981" opacity="0.22"/><rect x="192" y="44" width="92" height="76" rx="3" fill="none" stroke="#10b981" stroke-width="2"/></g>
  <g class="pc-s pc-s3"><rect x="288" y="44" width="92" height="76" rx="3" fill="#10b981" opacity="0.22"/><rect x="288" y="44" width="92" height="76" rx="3" fill="none" stroke="#10b981" stroke-width="2"/></g>
  <g class="pc-s pc-s4"><rect x="384" y="44" width="92" height="76" rx="3" fill="#10b981" opacity="0.22"/><rect x="384" y="44" width="92" height="76" rx="3" fill="none" stroke="#10b981" stroke-width="2"/></g>
  <g class="pc-s pc-sx">
    <rect x="480" y="44" width="92" height="76" rx="3" fill="#ef4444" opacity="0.18"/>
    <rect x="480" y="44" width="92" height="76" rx="3" fill="none" stroke="#ef4444" stroke-width="2"/>
    <rect x="576" y="44" width="92" height="76" rx="3" fill="none" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4 3"/>
    <text x="622" y="138" font-size="10" fill="#ef4444" text-anchor="middle">not reusable</text>
  </g>
  <text x="0" y="178" font-size="12" fill="currentColor">The tail blocks hold identical tokens, and their keys and values still differ, because they were</text>
  <text x="0" y="196" font-size="12" fill="currentColor">computed after a different question. Position, not content, decides what a block holds.</text>
  <text x="0" y="220" font-size="11.5" fill="currentColor" opacity="0.65">The whole prompt matters: chat template, system message, tool definitions, whitespace, tokenizer version.</text>
</svg>

## Hashing a block together with its history

A runtime needs to answer one question quickly. Given a new prompt, which of its blocks does the pool already hold?

vLLM hashes each full block, and it folds the parent block's hash into the child. From `hash_block_tokens` in `vllm/v1/core/kv_cache_utils.py`, the hash covers the parent hash, the block's token IDs, and a tuple of extra keys. Chaining is what makes the lookup correct. A block holding the same 64 tokens after a different prefix produces a different hash, so it cannot match by accident.

<svg viewBox="0 0 720 260" role="img" aria-label="Animated hash chain in which each block hash is computed from the previous hash and the block's token IDs" style="width:100%;height:auto;max-width:720px;">
  <style>
    @keyframes pc-h1 { 0%, 5% { opacity: 0 } 10%, 88% { opacity: 1 } 94%, 100% { opacity: 0 } }
    @keyframes pc-h2 { 0%, 25% { opacity: 0 } 30%, 88% { opacity: 1 } 94%, 100% { opacity: 0 } }
    @keyframes pc-h3 { 0%, 45% { opacity: 0 } 50%, 88% { opacity: 1 } 94%, 100% { opacity: 0 } }
    @keyframes pc-h4 { 0%, 65% { opacity: 0 } 70%, 88% { opacity: 1 } 94%, 100% { opacity: 0 } }
    .pc-h { opacity: 0 }
    .pc-h1 { animation: pc-h1 7s linear infinite }
    .pc-h2 { animation: pc-h2 7s linear infinite }
    .pc-h3 { animation: pc-h3 7s linear infinite }
    .pc-h4 { animation: pc-h4 7s linear infinite }
    @media (prefers-reduced-motion: reduce) { .pc-h { animation: none !important; opacity: 1 } }
  </style>
  <defs><marker id="pch" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Each block hash carries every block before it</text>
  <rect x="30" y="54" width="140" height="34" rx="3" fill="#3b82f6"/><text x="100" y="76" font-size="10" fill="#ffffff" text-anchor="middle">tokens 0-63</text>
  <rect x="206" y="54" width="140" height="34" rx="3" fill="#6366f1"/><text x="276" y="76" font-size="10" fill="#ffffff" text-anchor="middle">tokens 64-127</text>
  <rect x="382" y="54" width="140" height="34" rx="3" fill="#10b981"/><text x="452" y="76" font-size="10" fill="#ffffff" text-anchor="middle">tokens 128-191</text>
  <rect x="558" y="54" width="132" height="34" rx="3" fill="#f59e0b"/><text x="624" y="76" font-size="10" fill="#ffffff" text-anchor="middle">tokens 192-255</text>
  <g class="pc-h pc-h1"><text x="100" y="114" font-size="10.5" fill="currentColor" text-anchor="middle">h0 = hash(seed, tokens)</text></g>
  <g class="pc-h pc-h2">
    <line x1="174" y1="71" x2="200" y2="71" stroke="currentColor" stroke-width="1.3" marker-end="url(#pch)"/>
    <text x="276" y="114" font-size="10.5" fill="currentColor" text-anchor="middle">h1 = hash(h0, tokens)</text>
  </g>
  <g class="pc-h pc-h3">
    <line x1="350" y1="71" x2="376" y2="71" stroke="currentColor" stroke-width="1.3" marker-end="url(#pch)"/>
    <text x="452" y="114" font-size="10.5" fill="currentColor" text-anchor="middle">h2 = hash(h1, tokens)</text>
  </g>
  <g class="pc-h pc-h4">
    <line x1="526" y1="71" x2="552" y2="71" stroke="currentColor" stroke-width="1.3" marker-end="url(#pch)"/>
    <text x="624" y="114" font-size="10.5" fill="currentColor" text-anchor="middle">h3 = hash(h2, tokens)</text>
  </g>
  <text x="0" y="158" font-size="12" fill="currentColor">Looking up h2 asks one question: has anyone stored these 64 tokens, after exactly these 128 tokens?</text>
  <text x="0" y="180" font-size="12" fill="currentColor">That is why a lookup can walk forward and stop at the first miss.</text>
  <text x="0" y="212" font-size="12" fill="currentColor" opacity="0.8">The default hash is SHA-256, set by prefix_caching_hash_algo in vllm/config/cache.py. The seed for the</text>
  <text x="0" y="230" font-size="12" fill="currentColor" opacity="0.8">first block comes from PYTHONHASHSEED when set, and from a fixed string otherwise.</text>
</svg>

## The cache is the free list

There is no separate cache. The pool from the last article already has everything a cache needs.

When a request finishes, its blocks lose their reference and go on the free list, keeping their contents and their hash entry. A block in that state is free and cached at the same time. If a matching request arrives first, the block gets referenced again and nothing is recomputed. If the allocator needs space first, the block gets taken from the front of the list, its hash entry goes away, and the content is overwritten.

<svg viewBox="0 0 720 235" role="img" aria-label="A freed block stays on the LRU free list with its hash, and is either reused on a cache hit or overwritten when the allocator needs space" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pcf" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">One block, after its request finishes</text>
  <rect x="20" y="58" width="190" height="52" rx="4" fill="#3b82f6"/>
  <text x="115" y="80" font-size="11" fill="#ffffff" text-anchor="middle">ref_cnt drops to 0</text>
  <text x="115" y="97" font-size="11" fill="#ffffff" text-anchor="middle">hash and contents kept</text>
  <line x1="216" y1="84" x2="256" y2="70" stroke="currentColor" stroke-width="1.3" marker-end="url(#pcf)"/>
  <line x1="216" y1="90" x2="256" y2="148" stroke="currentColor" stroke-width="1.3" marker-end="url(#pcf)"/>
  <rect x="262" y="40" width="300" height="52" rx="4" fill="none" stroke="#10b981" stroke-width="1.4"/>
  <text x="276" y="62" font-size="11.5" font-weight="600" fill="#10b981">A matching request arrives first</text>
  <text x="276" y="80" font-size="11" fill="currentColor">the block is referenced again, nothing is recomputed</text>
  <rect x="262" y="124" width="300" height="52" rx="4" fill="none" stroke="#f59e0b" stroke-width="1.4"/>
  <text x="276" y="146" font-size="11.5" font-weight="600" fill="#f59e0b">The allocator needs space first</text>
  <text x="276" y="164" font-size="11" fill="currentColor">the hash entry is dropped and the content overwritten</text>
  <text x="0" y="206" font-size="12" fill="currentColor">The list is ordered least recently used first, so the oldest unreferenced block goes first.</text>
  <text x="0" y="226" font-size="12" fill="currentColor" opacity="0.8">Nothing is copied and nothing is written back. Caching here means not yet overwriting.</text>
</svg>

Two consequences follow. Prefix caching costs no extra memory, because the blocks were sitting there anyway. It also offers no guarantee, because any block can lose its contents the moment the pool is under pressure.

## Only full blocks get cached

`cache_full_blocks` in `vllm/v1/core/block_pool.py` stores full blocks. A trailing partial block has no hash and no entry, since appending the next token would change its contents.

With the default block size of 64, a shared prefix of 100 tokens caches one block. The other 36 tokens get prefilled on every request.

<svg viewBox="0 0 720 200" role="img" aria-label="A 100-token shared prefix at block size 64 caches only the first 64 tokens, leaving 36 to be recomputed each time" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">A 100-token shared prefix, block size 64</text>
  <rect x="40" y="48" width="320" height="34" rx="3" fill="#10b981"/>
  <text x="200" y="70" font-size="11" fill="#ffffff" text-anchor="middle">tokens 0-63, full block, cached</text>
  <rect x="364" y="48" width="180" height="34" rx="3" fill="currentColor" opacity="0.12"/>
  <rect x="364" y="48" width="180" height="34" rx="3" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="454" y="70" font-size="11" fill="currentColor" text-anchor="middle">tokens 64-99, partial</text>
  <text x="556" y="70" font-size="11" fill="#ef4444">prefilled every time</text>
  <text x="0" y="122" font-size="12" fill="currentColor">Block size sets the granularity of a hit. A long shared document loses at most 63 tokens at its tail,</text>
  <text x="0" y="140" font-size="12" fill="currentColor">which is noise. A short shared prefix can fall below one block and never cache at all.</text>
  <text x="0" y="172" font-size="12" fill="currentColor" opacity="0.8">If you depend on prefix hits for short prompts, a smaller block size buys finer matching, and the</text>
  <text x="0" y="190" font-size="12" fill="currentColor" opacity="0.8">costs from the last article still apply.</text>
</svg>

There is one more boundary case. If a prompt hits the cache completely, the scheduler still recomputes its last token. Sampling the first output token needs logits, and the cache stores keys and values rather than logits. `kv_cache_manager.py` caps the hit at `prompt_length - 1` for exactly this reason.

## One salted block isolates a tenant

Shared prefixes across tenants are a security question, not only a performance one. A request that returns quickly tells the sender that someone else recently sent that prefix. No KV bytes leak, and the timing does.

vLLM's control is `cache_salt`. A request carries a salt string, and the salt enters the extra keys of the first block only, at `start_token_idx == 0`. Because hashes chain, changing the first block's hash changes every hash after it.

<svg viewBox="0 0 720 250" role="img" aria-label="Two tenants send identical tokens, and because each salts the first block their entire hash chains differ" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Identical tokens, two tenants, no shared blocks</text>
  <text x="0" y="58" font-size="11" fill="currentColor">tenant A</text>
  <rect x="86" y="40" width="140" height="32" rx="3" fill="#3b82f6"/><text x="156" y="60" font-size="9.5" fill="#ffffff" text-anchor="middle">salt A + tokens 0-63</text>
  <rect x="232" y="40" width="140" height="32" rx="3" fill="#3b82f6" opacity="0.75"/><text x="302" y="60" font-size="9.5" fill="#ffffff" text-anchor="middle">tokens 64-127</text>
  <rect x="378" y="40" width="140" height="32" rx="3" fill="#3b82f6" opacity="0.55"/><text x="448" y="60" font-size="9.5" fill="#ffffff" text-anchor="middle">tokens 128-191</text>
  <text x="528" y="60" font-size="10" fill="currentColor" opacity="0.8">h0a, h1a, h2a</text>
  <text x="0" y="112" font-size="11" fill="currentColor">tenant B</text>
  <rect x="86" y="94" width="140" height="32" rx="3" fill="#f59e0b"/><text x="156" y="114" font-size="9.5" fill="#ffffff" text-anchor="middle">salt B + tokens 0-63</text>
  <rect x="232" y="94" width="140" height="32" rx="3" fill="#f59e0b" opacity="0.75"/><text x="302" y="114" font-size="9.5" fill="#ffffff" text-anchor="middle">tokens 64-127</text>
  <rect x="378" y="94" width="140" height="32" rx="3" fill="#f59e0b" opacity="0.55"/><text x="448" y="114" font-size="9.5" fill="#ffffff" text-anchor="middle">tokens 128-191</text>
  <text x="528" y="114" font-size="10" fill="currentColor" opacity="0.8">h0b, h1b, h2b</text>
  <text x="0" y="166" font-size="12" fill="currentColor">Only the first block carries the salt. The chain carries it the rest of the way, so every later hash differs</text>
  <text x="0" y="184" font-size="12" fill="currentColor">and the two tenants never read each other's blocks.</text>
  <text x="0" y="214" font-size="12" fill="currentColor" opacity="0.8">Pick the trust boundary deliberately. One salt for everyone allows the widest sharing. One salt per</text>
  <text x="0" y="232" font-size="12" fill="currentColor" opacity="0.8">tenant isolates tenants and keeps sharing inside each one. One salt per request turns the cache off.</text>
</svg>

## What improves, and by how much

A hit removes prefill work for the matched positions. It removes nothing else.

<svg viewBox="0 0 720 215" role="img" aria-label="A cache hit shortens the prefill portion of a request and leaves the decode portion unchanged" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Where the time goes, before and after a hit</text>
  <text x="0" y="60" font-size="11.5" fill="currentColor">miss</text>
  <rect x="80" y="44" width="300" height="26" rx="2" fill="#3b82f6"/><text x="230" y="62" font-size="10" fill="#ffffff" text-anchor="middle">prefill</text>
  <rect x="382" y="44" width="280" height="26" rx="2" fill="#8b5cf6"/><text x="522" y="62" font-size="10" fill="#ffffff" text-anchor="middle">decode</text>
  <text x="0" y="112" font-size="11.5" fill="currentColor">hit</text>
  <rect x="80" y="96" width="46" height="26" rx="2" fill="#3b82f6"/>
  <rect x="128" y="96" width="252" height="26" rx="2" fill="currentColor" opacity="0.1"/>
  <text x="254" y="114" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.7">skipped</text>
  <rect x="382" y="96" width="280" height="26" rx="2" fill="#8b5cf6"/><text x="522" y="114" font-size="10" fill="#ffffff" text-anchor="middle">decode, unchanged</text>
  <line x1="382" y1="38" x2="382" y2="128" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>
  <text x="388" y="34" font-size="10" fill="currentColor" opacity="0.7">first output token</text>
  <text x="0" y="162" font-size="12" fill="currentColor">Time to first token falls. Time per output token does not, because decode has the same work to do.</text>
  <text x="0" y="184" font-size="12" fill="currentColor" opacity="0.8">Under load the saved prefill also frees the GPU for other requests, which shortens queues.</text>
</svg>

The number to track is matched tokens per request, not hit rate. A request that matches 16 tokens and one that matches 4,000 both count as hits, and they are not worth the same. vLLM reports `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` in tokens, so the ratio of those two already says what you want.

For the Llama 3.1 70B example at 320 KiB per token, a 4,096-token document holds 1.25 GiB of KV across the model. Keeping it resident is free while nothing else needs those blocks, and it costs you a smaller working set the moment something does.

## A warm cache lives on one replica

Prefix caching is local to a GPU pool. A prefix that is warm on replica A does nothing for replica B.

<svg viewBox="0 0 720 215" role="img" aria-label="Load balancing sends identical prompts to different replicas and misses, while cache-aware routing sends them to the replica that holds the prefix" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="pcr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>
  <text x="0" y="18" font-size="13" font-weight="600" fill="currentColor">Three requests with the same prefix</text>
  <text x="0" y="56" font-size="11" fill="currentColor" opacity="0.8">round robin</text>
  <rect x="100" y="42" width="80" height="26" rx="3" fill="currentColor" opacity="0.12"/><text x="140" y="59" font-size="10" fill="currentColor" text-anchor="middle">router</text>
  <line x1="184" y1="55" x2="236" y2="44" stroke="currentColor" stroke-width="1.2" marker-end="url(#pcr)"/>
  <line x1="184" y1="55" x2="236" y2="55" stroke="currentColor" stroke-width="1.2" marker-end="url(#pcr)"/>
  <line x1="184" y1="55" x2="236" y2="66" stroke="currentColor" stroke-width="1.2" marker-end="url(#pcr)"/>
  <rect x="242" y="32" width="110" height="22" rx="3" fill="#ef4444" opacity="0.85"/><text x="297" y="47" font-size="9.5" fill="#ffffff" text-anchor="middle">replica A, miss</text>
  <rect x="242" y="56" width="110" height="22" rx="3" fill="#ef4444" opacity="0.85"/><text x="297" y="71" font-size="9.5" fill="#ffffff" text-anchor="middle">replica B, miss</text>
  <rect x="242" y="80" width="110" height="22" rx="3" fill="#ef4444" opacity="0.85"/><text x="297" y="95" font-size="9.5" fill="#ffffff" text-anchor="middle">replica C, miss</text>
  <text x="366" y="71" font-size="11" fill="currentColor">three full prefills, three copies of the same KV</text>
  <text x="0" y="146" font-size="11" fill="currentColor" opacity="0.8">prefix aware</text>
  <rect x="100" y="132" width="80" height="26" rx="3" fill="currentColor" opacity="0.12"/><text x="140" y="149" font-size="10" fill="currentColor" text-anchor="middle">router</text>
  <line x1="184" y1="145" x2="236" y2="145" stroke="currentColor" stroke-width="1.2" marker-end="url(#pcr)"/>
  <rect x="242" y="134" width="110" height="22" rx="3" fill="#10b981"/><text x="297" y="149" font-size="9.5" fill="#ffffff" text-anchor="middle">replica A, hit</text>
  <text x="366" y="149" font-size="11" fill="currentColor">one prefill, one copy, and replica A carries the load</text>
  <text x="0" y="200" font-size="12" fill="currentColor">Routing for cache hits concentrates traffic. Routing for even load destroys every hit.</text>
</svg>

Routing by prefix raises the hit rate and skews the load, so a router has to weigh cache affinity against queue depth on the replica that holds the prefix. Sending everything to the warm replica is how you turn a cache win into a latency regression.

## When it does not pay

Short prompts with no shared opening cannot hit. A template that changes per request breaks the chain at its first block, which makes every later block unmatchable. A timestamp or a user name near the front does it. Put the variable part at the end of the prompt, after the stable part, and the stable part stays cacheable.

Retained blocks also compete with running sequences. The free list holds cached blocks only until the allocator wants them, so a server under memory pressure evicts prefixes before it preempts requests. A high hit rate measured on an idle server tells you very little about a busy one.

## Summary

- Prefix caching skips the prefill for prompt tokens that an earlier request already computed. It never skips decode.
- A match is exact on token IDs from position zero, and it stops for good at the first difference.
- vLLM hashes each full block together with its parent's hash, so a block matches only after an identical history.
- The cache is the free list. Freed blocks keep their contents and hash until the allocator needs the space, so the feature costs nothing and guarantees nothing.
- Only full blocks are cached, and a complete hit still recomputes the last token to get logits.
- `cache_salt` enters the first block's hash and the chain carries it, so one salted block isolates a tenant.
- Hits improve time to first token. Routing decides how often a hit is even possible.

Next in this series: chunked prefill, and what to do about the long prompt that misses.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv][paged]
- vLLM source, in `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/block_pool.py`, and `vllm/v1/core/kv_cache_manager.py`
- [vLLM automatic prefix caching design][vapc], [production metrics][vmetrics], and [security guidance][vsec] on the timing side channel and `cache_salt`
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS 2024, which indexes shared prefixes with a radix tree instead of a hash chain. [paper][sglang]
- [Meta Llama 3.1 model configuration][llama], the source of the 320 KiB per token figure

[paged]: https://arxiv.org/abs/2309.06180
[vapc]: https://docs.vllm.ai/en/latest/design/prefix_caching/
[vmetrics]: https://docs.vllm.ai/en/latest/usage/metrics/
[vsec]: https://docs.vllm.ai/en/latest/usage/security/
[sglang]: https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf
[llama]: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
