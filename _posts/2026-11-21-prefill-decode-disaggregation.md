---
layout: post
title: "LLM Inference #9: Prefill–Decode Disaggregation"
date: 2026-11-21 00:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, disaggregation, kv-cache, scheduling, latency]
description: "Three users are streaming when a fourth sends a 4,096-token prompt. See why their next tokens wait and what it costs to give prompt processing its own GPUs."
---

Three users, A, B, and C, are halfway through their answers. Then D sends a 4,096-token document. What happens to A's next token?

## A's next token waits for the whole batch

A, B, and C are in **decode**. Each GPU iteration generates one more token for each stream. D needs **prefill** first. The GPU processes D's prompt and saves the attention keys and values that later steps will reuse. That saved state is the **KV cache**.

When one iteration does both jobs, A gets its next token when the batch finishes. If the batch includes all of D's prompt, that can make one long gap. [Chunked prefill]({% post_url 2026-11-14-chunked-prefill %}) spreads D's prompt across several iterations. A can get a token after each chunk, but a mixed iteration can still take longer than decode alone. Smaller chunks can shorten those gaps while making D wait through more iterations.

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 360 230" role="img" aria-labelledby="pd-shared-title" aria-describedby="pd-shared-desc" style="display:block;width:100%;height:auto;max-width:540px;">
  <title id="pd-shared-title">When A gets its next token on shared GPUs</title>
  <desc id="pd-shared-desc">Time runs left to right. With decode alone, A receives a token after each short iteration. Adding D's whole prompt to one iteration creates one long gap. Splitting D's prompt allows A to receive tokens sooner, though mixed iterations can still be longer than decode alone.</desc>
  <rect x="1" y="1" width="358" height="228" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <g fill="currentColor" font-size="14" font-weight="600">
    <text x="18" y="29">Decode alone</text>
    <text x="18" y="91">Decode with D's whole prompt</text>
    <text x="18" y="153">Decode with D's prompt in chunks</text>
  </g>
  <g fill="#10b981" fill-opacity="0.2" stroke="#10b981">
    <rect x="18" y="38" width="58" height="18" rx="3"/>
    <rect x="76" y="38" width="58" height="18" rx="3"/>
    <rect x="134" y="38" width="58" height="18" rx="3"/>
    <rect x="192" y="38" width="58" height="18" rx="3"/>
    <rect x="250" y="38" width="58" height="18" rx="3"/>
    <rect x="250" y="100" width="58" height="18" rx="3"/>
  </g>
  <g fill="#8b5cf6" fill-opacity="0.18" stroke="#8b5cf6">
    <rect x="18" y="100" width="232" height="18" rx="3"/>
    <rect x="18" y="162" width="96" height="18" rx="3"/>
    <rect x="114" y="162" width="96" height="18" rx="3"/>
    <rect x="210" y="162" width="96" height="18" rx="3"/>
  </g>
  <g fill="currentColor">
    <circle cx="18" cy="56" r="3.5"/><circle cx="76" cy="56" r="3.5"/><circle cx="134" cy="56" r="3.5"/>
    <circle cx="192" cy="56" r="3.5"/><circle cx="250" cy="56" r="3.5"/><circle cx="308" cy="56" r="3.5"/>
    <circle cx="18" cy="118" r="3.5"/><circle cx="250" cy="118" r="3.5"/><circle cx="308" cy="118" r="3.5"/>
    <circle cx="18" cy="180" r="3.5"/><circle cx="114" cy="180" r="3.5"/><circle cx="210" cy="180" r="3.5"/><circle cx="306" cy="180" r="3.5"/>
    <circle cx="18" cy="209" r="3.5"/>
    <text x="29" y="213" font-size="13">A gets a token</text>
    <text x="279" y="213" font-size="13">Time</text>
  </g>
  <path d="M311 209 H339 M334 205 L339 209 L334 213" fill="none" stroke="currentColor" stroke-width="1.3"/>
</svg>
<figcaption style="font-size:0.875rem;margin-top:0.5rem;">Each bar is one iteration. Purple bars include prompt work. The timing is illustrative.</figcaption>
</figure>

## DistServe puts prefill and decode on separate GPUs
{: #distserve-gives-each-phase-its-own-gpus}

DistServe gives prompt processing its own pool of GPUs. A, B, and C can keep decoding while D's prompt runs because they no longer share an iteration. You can also size each pool for its own work. The new cost is that D's cache has to reach the decode pool. [The DistServe paper on separating the two phases][distserve-design]

DistServe's prefill worker builds the prompt KV and samples D's first output token. The decode worker pulls the cache when it needs it and receives that token too. Its next model step uses token 1 and the prompt KV to produce token 2. This describes what the workers exchange. When the client sees token 1 depends on how the service sends it. [DistServe's decode handoff][distserve-handoff] · [DistServe's pull scheduling][distserve-pull]

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 360 164" role="img" aria-labelledby="pd-split-title" aria-describedby="pd-split-desc" style="display:block;width:100%;height:auto;max-width:510px;">
  <title id="pd-split-title">DistServe's handoff between GPU pools</title>
  <desc id="pd-split-desc">The prefill pool processes D's prompt and samples output token 1. The decode pool can serve A, B, and C in parallel. It pulls D's prompt cache and receives token 1. Its next step for D produces token 2.</desc>
  <rect x="1" y="1" width="358" height="162" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <rect x="16" y="15" width="328" height="46" rx="5" fill="#8b5cf6" fill-opacity="0.15" stroke="#8b5cf6"/>
  <rect x="16" y="103" width="328" height="46" rx="5" fill="#10b981" fill-opacity="0.15" stroke="#10b981"/>
  <path d="M46 67 V96 M42 91 L46 96 L50 91" fill="none" stroke="#3b82f6" stroke-width="1.6"/>
  <g fill="currentColor" font-size="14">
    <text x="28" y="34" font-weight="600">Prefill pool</text>
    <text x="28" y="51">Builds D's KV and samples token 1</text>
    <text x="59" y="87">Prompt KV + token 1</text>
    <text x="28" y="122" font-weight="600">Decode pool</text>
    <text x="28" y="139">Streams A, B, and C. Makes token 2 for D.</text>
  </g>
</svg>
</figure>

## How a vLLM proxy joins the two pools
{: #follow-d-through-vllm}

vLLM also supports running prefill and decode on separate workers. [vLLM's guide to disaggregated prefill][vllm]

The client still wants to send one request and get one stream back. In one vLLM example, a **proxy** is the front door that coordinates those two workers. It sends D's prompt to prefill and gets back the cache locations. Then it sends the original completion request and those locations to decode. The decode worker pulls the KV directly from prefill, and the proxy sends decode's stream back to the client.

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 360 271" role="img" aria-labelledby="pd-proxy-title" aria-describedby="pd-proxy-desc" style="display:block;width:100%;height:auto;max-width:500px;">
  <title id="pd-proxy-title">One request through a vLLM proxy</title>
  <desc id="pd-proxy-desc">The client sends a request to the proxy. The proxy sends the prompt to prefill, gets the cache locations, and forwards the original request and those locations to decode. The workers move the KV directly. Decode streams the answer through the proxy to the client.</desc>
  <defs><marker id="pd-proxy-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="context-stroke"/></marker></defs>
  <rect x="1" y="1" width="358" height="269" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <g fill="currentColor" font-size="14" font-weight="600" text-anchor="middle">
    <text x="34" y="27">Client</text>
    <text x="115" y="27">Proxy</text>
    <text x="229" y="27">Prefill</text>
    <text x="326" y="27">Decode</text>
  </g>
  <path d="M34 36 V255 M115 36 V255 M229 36 V140 M229 158 V196 M326 36 V255" fill="none" stroke="currentColor" stroke-opacity="0.25" stroke-dasharray="3 4"/>
  <g fill="none" stroke-width="1.5" marker-end="url(#pd-proxy-arrow)">
    <path d="M35 59 H112" stroke="currentColor"/>
    <path d="M116 90 H226" stroke="#8b5cf6"/>
    <path d="M228 121 H118" stroke="#3b82f6"/>
    <path d="M116 154 H323" stroke="#3b82f6"/>
    <path d="M230 189 H323" stroke="#3b82f6"/>
    <path d="M325 221 H118" stroke="#10b981"/>
    <path d="M114 252 H37" stroke="#10b981"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle">
    <text x="74" y="53">Request</text>
    <text x="172" y="84">Prompt</text>
    <text x="172" y="115">Cache locations</text>
    <text x="220" y="147">Request + cache locations</text>
    <text x="278" y="183">KV cache</text>
    <text x="220" y="215">Answer stream</text>
    <text x="74" y="246">Answer</text>
  </g>
</svg>
</figure>

### Why decode runs the last prompt position again
{: #why-vllm-reruns-one-prompt-position}

In this example, decode handles the entire client response through its usual completion path. Prefill samples a token for its internal reply. The proxy passes the original request and the cache locations to decode, so that internal sample stays out of the client stream.

The cache holds keys and values. The scores used to choose the next token are a separate result of running the model. When decode can reuse the KV for the entire prompt, it runs just the last prompt position again to recover those scores. Decode then samples the first token the client sees. D waits for the cache transfer and that extra step before seeing an answer.

## How big is D's cache?
{: #how-much-kv-moves}

Take the 70B Llama example from [Part 1 on the KV cache]({% post_url 2026-09-23-llm-inference-kv-cache-prefill-decode %}). With full attention, every prompt position has a key and a value for each of eight KV heads in each of 80 layers. Each key or value has 128 elements, and BF16 uses two bytes per element. That works out to 320 KiB per position.

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 360 91" role="img" aria-labelledby="pd-size-title" aria-describedby="pd-size-desc" style="display:block;width:100%;height:auto;max-width:510px;">
  <title id="pd-size-title">D's prompt holds 1.25 GiB of logical KV</title>
  <desc id="pd-size-desc">With full attention and BF16 cache values, one position uses 320 KiB. Multiply by 4,096 prompt positions to get 1.25 GiB for one copy across the model.</desc>
  <rect x="1" y="1" width="358" height="89" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <rect x="250" y="14" width="97" height="62" rx="5" fill="#3b82f6" fill-opacity="0.15" stroke="#3b82f6"/>
  <g fill="currentColor" text-anchor="middle">
    <g font-size="17" font-weight="600">
      <text x="60" y="41">320 KiB</text>
      <text x="118" y="41">×</text>
      <text x="176" y="41">4,096</text>
      <text x="232" y="41">=</text>
      <text x="298" y="41">1.25 GiB</text>
    </g>
    <g font-size="13">
      <text x="60" y="61">per position</text>
      <text x="176" y="61">positions</text>
      <text x="298" y="61">prompt KV</text>
    </g>
  </g>
</svg>
<figcaption style="font-size:0.875rem;margin-top:0.5rem;">Calculated for one copy of D's prompt KV across the model.</figcaption>
</figure>

Actual network traffic depends on where the workers run and which blocks decode can reuse. If one 100 Gb/s link carries the whole payload once at its ideal rate, the bytes alone take about 107 ms. That calculation excludes setup, queueing, and contention.

## Does the split pay off?
{: #measure-the-wait-and-the-gaps}

D could wait longer for its first token while A, B, and C see shorter gaps. D could also get a first token sooner if dedicated prefill saves more time than the handoff costs. A fixed split can leave one pool idle while the other builds a queue.

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 360 124" role="img" aria-labelledby="pd-latency-title" aria-describedby="pd-latency-desc" style="display:block;width:100%;height:auto;max-width:540px;">
  <title id="pd-latency-title">First-token wait and gaps during a stream</title>
  <desc id="pd-latency-desc">On a timeline of one response, time to first token spans the request and the first output token. Each gap after that spans two consecutive output tokens. The lengths are illustrative.</desc>
  <rect x="1" y="1" width="358" height="122" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <path d="M21 55 H339 M334 51 L339 55 L334 59" fill="none" stroke="currentColor" stroke-opacity="0.65" stroke-width="1.3"/>
  <g fill="currentColor" font-size="13">
    <text x="21" y="32">Send request</text>
    <text x="181" y="32" text-anchor="middle">Token 1</text>
    <text x="252" y="32" text-anchor="middle">Token 2</text>
    <text x="322" y="32" text-anchor="middle">Token 3</text>
    <circle cx="21" cy="55" r="3.5"/><circle cx="181" cy="55" r="3.5"/>
    <circle cx="252" cy="55" r="3.5"/><circle cx="322" cy="55" r="3.5"/>
    <text x="101" y="104" text-anchor="middle">Time to first token</text>
    <text x="216.5" y="104" text-anchor="middle">Token gap</text>
    <text x="287" y="104" text-anchor="middle">Token gap</text>
  </g>
  <path d="M21 71 V82 H181 V71" fill="none" stroke="#8b5cf6" stroke-width="1.6"/>
  <path d="M181 71 V82 H252 V71 M252 71 V82 H322 V71" fill="none" stroke="#10b981" stroke-width="1.6"/>
</svg>
<figcaption style="font-size:0.875rem;margin-top:0.5rem;">Measure both intervals at the client. The timeline shows their boundaries.</figcaption>
</figure>

Replay the same requests at the same arrival times on the same hardware. Tune the chunk size for shared GPUs and the pool sizes for the split. Compare the p99 time to first token for new requests like D and the p99 token gap for streams already running. At p99, about 99 out of 100 waits are at or below the reported value. Include the gap from token 1 to token 2, especially if the service can send token 1 before the handoff finishes.

If chunking meets both targets at your busiest load, it already does the job. Use the split if it brings the stream gaps within target and keeps the first-token wait within budget.

[distserve-design]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5
[distserve-handoff]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=6
[distserve-pull]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=9
[vllm]: https://docs.vllm.ai/en/latest/features/disagg_prefill/
