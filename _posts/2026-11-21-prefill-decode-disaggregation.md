---
layout: post
title: "LLM Inference #9: Prefill–Decode Disaggregation"
date: 2026-11-21 00:00:00 -0500
categories: [AI, LLM Inference]
tags: [llm, inference, disaggregation, kv-cache, scheduling, latency]
description: "Three users are streaming when a fourth sends a 4,096-token prompt. Follow its KV cache across two GPU pools and see when the handoff is worth it."
---

Three users, A, B, and C, are halfway through their answers. A fourth user, D, sends a 4,096-token document. D has no reusable prefix, and there is room for its KV cache. These are the same four requests from [Chunked Prefill]({% post_url 2026-11-14-chunked-prefill %}).

Prefill processes D's prompt and saves the keys and values that attention will need again. The saved state is the **KV cache**. Decode uses that cache as it generates the answer. What changes if we give prefill and decode separate GPUs? [KV Cache, Prefill, and Decode]({% post_url 2026-09-23-llm-inference-kv-cache-prefill-decode %})

## DistServe gives each phase its own GPUs

DistServe sends D's prompt to a prefill pool. A, B, and C keep using a separate decode pool, so none of their iterations have to process D's prompt. D's KV has to reach the decode worker before D can continue there. This split is **prefill–decode disaggregation**. [DistServe, §2.3][distserve-architecture]

Recall what happens when both phases share the GPUs. With a 1,024-token budget, A, B, and C each take one input position per iteration. That leaves D up to 1,021. If the three streams stay active and their order holds, D needs four chunks of 1,021 and a final chunk of 12. The streams get a token each turn, after the prompt chunk in that turn finishes. [Sarathi-Serve][sarathi]

<figure style="margin:1.5rem 0;">
<div style="display:flex;flex-wrap:wrap;gap:1rem;">
<div style="flex:1 1 16rem;min-width:0;">
<svg viewBox="0 0 324 164" role="img" aria-labelledby="pd-shared-title" aria-describedby="pd-shared-desc" style="display:block;width:100%;height:auto;">
  <title id="pd-shared-title">Shared GPU pool</title>
  <desc id="pd-shared-desc">Each iteration batches a prompt chunk from D with a decode step for A, B, and C. D keeps its cache on the same worker.</desc>
  <rect x="1" y="1" width="322" height="162" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <text x="13" y="27" font-size="17" font-weight="600" fill="currentColor">Shared GPU pool</text>

  <g fill="currentColor" fill-opacity="0.06">
    <rect x="13" y="43" width="137" height="83" rx="4"/>
    <rect x="174" y="43" width="137" height="83" rx="4"/>
  </g>
  <g fill="#8b5cf6" fill-opacity="0.16" stroke="#8b5cf6">
    <rect x="19" y="65" width="125" height="23" rx="4"/>
    <rect x="180" y="65" width="125" height="23" rx="4"/>
  </g>
  <g fill="#10b981" fill-opacity="0.16" stroke="#10b981">
    <rect x="19" y="95" width="125" height="23" rx="4"/>
    <rect x="180" y="95" width="125" height="23" rx="4"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle">
    <text x="82" y="58">Iteration 1</text>
    <text x="243" y="58">Iteration 2</text>
    <text x="82" y="81">D's prompt chunk</text>
    <text x="243" y="81">D's prompt chunk</text>
    <text x="82" y="111">Decode A, B, C</text>
    <text x="243" y="111">Decode A, B, C</text>
  </g>
  <path d="M155 85 H168 M164 81 L168 85 L164 89" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="13" y="148" font-size="13" fill="currentColor">D's KV stays on the worker.</text>
</svg>
</div>
<div style="flex:1 1 16rem;min-width:0;">
<svg viewBox="0 0 324 164" role="img" aria-labelledby="pd-split-title" aria-describedby="pd-split-desc" style="display:block;width:100%;height:auto;">
  <title id="pd-split-title">Separate GPU pools</title>
  <desc id="pd-split-desc">The prefill pool processes D's prompt while the decode pool serves A, B, and C. D hands its KV to the decode pool and joins when ready.</desc>
  <rect x="1" y="1" width="322" height="162" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <text x="13" y="27" font-size="17" font-weight="600" fill="currentColor">Separate GPU pools</text>

  <rect x="13" y="43" width="298" height="32" rx="4" fill="#8b5cf6" fill-opacity="0.16" stroke="#8b5cf6"/>
  <rect x="13" y="105" width="298" height="32" rx="4" fill="#10b981" fill-opacity="0.16" stroke="#10b981"/>
  <path d="M149 79 V100 M145 96 L149 100 L153 96" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
  <g fill="currentColor" font-size="13">
    <text x="24" y="63" font-weight="600">Prefill pool</text>
    <text x="123" y="63">Process D's prompt</text>
    <text x="162" y="94">D's KV moves</text>
    <text x="24" y="125" font-weight="600">Decode pool</text>
    <text x="123" y="125">A, B, C stream. D joins later.</text>
  </g>
</svg>
</div>
</div>
<figcaption style="font-size:0.875rem;margin-top:0.5rem;">The boxes show where work runs. Both pools in the split can work at the same time.</figcaption>
</figure>

### DistServe hands off the first token too

At the end of prefill, the model has scores for the next token. DistServe uses those scores to choose the first output token on the prefill GPU. The decode worker gets that token and the prompt KV. It feeds the first output token back through the model, attends to the prompt KV, and produces token 2. DistServe describes this worker-to-worker handoff. When the client sees token 1 depends on how the service delivers output. [DistServe, §3.1 and §3.2][distserve-phases]

When requests come in bursts, moving every finished cache to decode at once can fill its memory. DistServe lets the decode worker **pull** KV when it needs it. The completed cache waits in prefill GPU memory until then. That gives decode control over when it takes the cache. It also means a backlog can occupy space on the prefill side. [DistServe, §4.3][distserve-scheduling]

## Follow D through vLLM

This walkthrough uses [vLLM V1 at revision `a16bfee`][revision], the same revision as part 8. The NIXL accuracy test pairs [`toy_proxy_server.py`][nixl-launch] with [`NixlConnector`][nixl-config], an alias for [`NixlPullConnector`][nixl-alias]. Follow a streaming `/v1/completions` request with full attention, cold caches, speculation off, and a successful transfer.

<figure style="margin:1.5rem 0;">
<svg viewBox="0 0 344 354" role="img" aria-labelledby="pd-nixl-title" aria-describedby="pd-nixl-desc" style="display:block;width:100%;height:auto;max-width:480px;">
  <title id="pd-nixl-title">One request through the vLLM NIXL pull proxy</title>
  <desc id="pd-nixl-desc">The client sends a prompt to the proxy. Prefill samples one output token internally and returns cache locations. The proxy forwards the prompt and locations to decode. Decode reads the cache directly from prefill, reruns the last prompt position, and streams output to the client through the proxy.</desc>
  <defs><marker id="pd-nixl-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="context-stroke"/></marker></defs>
  <rect x="1" y="1" width="342" height="352" rx="8" fill="currentColor" fill-opacity="0.035" stroke="currentColor" stroke-opacity="0.2"/>
  <g fill="currentColor" font-size="14" font-weight="600" text-anchor="middle">
    <text x="30" y="26">Client</text>
    <text x="103" y="26">Proxy</text>
    <text x="212" y="26">Prefill</text>
    <text x="292" y="26">Decode</text>
  </g>
  <path d="M30 35 V337 M103 35 V337 M212 35 V109 M212 134 V176 M212 195 V246 M292 35 V256 M292 291 V337" fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 4"/>
  <g fill="none" stroke-width="1.5" marker-end="url(#pd-nixl-arrow)">
    <path d="M31 63 H100" stroke="currentColor"/>
    <path d="M104 102 H209" stroke="#8b5cf6"/>
    <path d="M211 153 H106" stroke="#3b82f6"/>
    <path d="M104 194 H289" stroke="#3b82f6"/>
    <path d="M213 236 H289" stroke="#3b82f6"/>
    <path d="M291 307 H106" stroke="#10b981"/>
    <path d="M102 334 H33" stroke="#10b981"/>
  </g>
  <g fill="currentColor" font-size="12" text-anchor="middle">
    <text x="66" y="57">Prompt</text>
    <text x="158" y="96">Prompt, cap at 1</text>
    <text x="158" y="147">KV locations</text>
    <text x="205" y="188">Prompt + KV locations</text>
    <text x="252" y="230">Pull KV blocks</text>
    <text x="205" y="301">Stream from decode</text>
    <text x="66" y="328">Output</text>
  </g>
  <rect x="159" y="109" width="106" height="25" rx="4" fill="#8b5cf6" fill-opacity="0.16" stroke="#8b5cf6"/>
  <text x="212" y="125" fill="currentColor" font-size="12" text-anchor="middle">Samples 1 token</text>
  <rect x="247" y="256" width="90" height="35" rx="4" fill="#10b981" fill-opacity="0.16" stroke="#10b981"/>
  <text x="292" y="270" fill="currentColor" font-size="12" text-anchor="middle">Reruns last</text>
  <text x="292" y="284" fill="currentColor" font-size="12" text-anchor="middle">prompt position</text>
</svg>
<figcaption style="font-size:0.875rem;margin-top:0.5rem;">The proxy passes cache locations. The KV bytes go directly from prefill to decode.</figcaption>
</figure>

1. **The proxy asks prefill for one token.** It picks a prefill server in round-robin order, turns streaming off for that call, and caps the reply at one token. Prefill processes the prompt and samples a token for the internal reply. The connector keeps the cache blocks alive for pickup and returns `kv_transfer_params`. That metadata identifies the source engine, request, block IDs, and where decode can contact prefill. The proxy sends the original prompt and that metadata to the next decode server in its rotation. The prefill token stays out of the client stream. [Prefill call][proxy-prefill] · [Handoff metadata][nixl-handoff] · [Proxy routing][proxy]

2. **Decode reserves blocks before the copy.** The scheduler asks the connector how much prompt KV is available remotely. It reserves space for that KV in the decode worker's cache, then marks D as waiting for remote KV. The connector gives the worker the source and destination block IDs it needs for the copy. D uses no model compute while it is waiting for the transfer. [Remote match][nixl-match] · [Scheduler allocation and wait][scheduler-admit] · [Worker transfer plan][nixl-plan]

3. **The decode worker pulls the bytes.** The worker starts an asynchronous NIXL `READ` into the destination blocks. The worker reports completed receives to the scheduler, which can then let D compete for a batch slot. A, B, and C can keep using decode iterations while D waits. [NIXL read][nixl-read] · [Worker completion report][worker-kv] · [Scheduler resume][scheduler-resume]

4. **Decode reruns the last prompt position and starts the stream.** With all 4,096 prompt positions loaded, the scheduler sets D's computed count back to 4,095. Decode runs that last position again and samples the first client-facing token. The completion server streams the output as server-sent events, and the proxy forwards that stream to D. [Last-position replay][scheduler-resume] · [Completion stream][completion-stream] · [Proxy stream][proxy]

## Why vLLM reruns one prompt position

Both examples let decode pull the cache. DistServe also hands decode the first output token, so its next model step produces token 2. In the vLLM example, decode gets the original completion request plus the cache locations. The prefill sample stays with the internal reply, and the KV bytes move directly between workers. [DistServe, §3.2][distserve-phases] · [vLLM proxy routing][proxy]

The cache holds attention keys and values. The scores used to choose token 1 come from running the last prompt position. Since the decode request carries neither those scores nor the prefill sample, decode reruns that one position to get the scores and sample token 1 itself. The tradeoff is one extra position of compute in exchange for using the ordinary decode-side completion path. Carrying the prefill sample forward would require the decode request to accept that token and resume generation from it. [Scheduler replay][scheduler-resume] · [Decode completion path][completion-generate]

For D, **time to first token (TTFT)** runs from sending the request to receiving its first output token. With this proxy, D waits for prefill, the cache handoff, and the first decode-side step. A service that emits the prefill token early can shift part of the handoff wait into the gap between tokens 1 and 2. Measure when the client gets output as well as when each worker produces it.

## How much KV moves?

With BF16 KV and full attention, Llama 3.1 70B keeps one key and one value for each of its eight KV heads in each of 80 layers. A head has 128 elements, and each BF16 element takes two bytes. [Meta's Llama 3.1 70B configuration][llama-config]

- One position uses 2 × 80 × 8 × 128 × 2 bytes, or **320 KiB**.
- D's 4,096 prompt positions use 4,096 × 320 KiB, or **1.25 GiB**.

That is one logical copy of the prompt KV across the whole model. The count covers the tensor values for those positions. Physical traffic depends on the cache layout and placement. The two pools can use different GPUs in the same host. Across hosts, the data may be split across links. A destination with reusable blocks can avoid copying those blocks. [DistServe, §3.3 and §4.2][distserve-bandwidth] · [NIXL local-cache reuse][nixl-local-hit]

If one 100 Gb/s path carries the entire payload once at its ideal rate of 12.5 GB/s, the bytes alone take about **107 ms**. This is a link calculation. Setup, contention, and extra copies affect the actual handoff time.

Splitwise shows how to overlap part of the transfer with prefill. It can start sending one layer's cache while prefill computes the next layer. [Splitwise, §IV-C][splitwise]

## Measure the wait and the gaps

D can get its first token sooner if dedicated prefill saves more time than the handoff adds. It can also wait longer while A, B, and C see shorter gaps. Pool sizes matter. With a fixed split, one pool can build a queue while the other has idle GPUs. Each pool also keeps its own copy of the weights, possibly sharded across several GPUs. [DistServe, §2.3 and §3][distserve-architecture]

Replay the same arrivals on both setups with the same GPU count, GPU type, and cluster topology. Use the same model and KV precision, and warm the caches under the same conditions. Keep the same mix of prompt and output lengths, including bursts of long prompts while users are already streaming. Tune the chunk budget for shared GPUs and the pool sizes for the split.

- Measure **p99 TTFT** by prompt length. That shows how long short requests and requests like D wait for the first token.
- Measure **p99 inter-token latency** over gaps between consecutive output tokens. Look at the streams already running when long prompts arrive. Check the gap from token 1 to token 2 separately. If a stream event bundles several tokens, report event gaps separately from token timings.
- Time the prefill queue, prompt processing, KV transfer, and decode admission. That tells you which part of the handoff D waited for.

Keep chunking if it meets both targets at the offered load. Test a split if long prompts still cause slow streams, or if chunks small enough to protect those streams make first tokens arrive too late. The split is worth using if it fixes the token gaps and still meets the first-token target on the same hardware.

DistServe measures a related outcome. At a chosen target such as 90% of requests meeting **both** their TTFT and average time per output token limits, its **per-GPU goodput** is the highest request arrival rate the system sustains, divided by the GPU count. Those are per-request limits. Track p99 token gaps separately to see pauses that a request's average can hide. [DistServe, §1 and §6.1][distserve-paper]

## References

- Zhong et al., [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*][distserve], OSDI 2024.
- Agrawal et al., [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*][sarathi], OSDI 2024.
- Patel et al., [*Splitwise: Efficient Generative LLM Inference Using Phase Splitting*][splitwise], ISCA 2024.
- [vLLM's disaggregated prefill guide][vllm].

[distserve]: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
[distserve-paper]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
[distserve-architecture]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=4
[distserve-phases]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5
[distserve-bandwidth]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=7
[distserve-scheduling]: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=9
[sarathi]: https://www.usenix.org/conference/osdi24/presentation/agrawal
[splitwise]: https://arxiv.org/pdf/2311.18677v2#page=7
[vllm]: https://docs.vllm.ai/en/latest/features/disagg_prefill/
[llama-config]: https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
[revision]: https://github.com/vllm-project/vllm/tree/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576
[nixl-launch]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh#L325-L338
[nixl-config]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh#L54-L60
[nixl-alias]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py#L455-L456
[proxy]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py#L213-L250
[proxy-prefill]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py#L153-L187
[nixl-handoff]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py#L251-L295
[nixl-match]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py#L59-L65
[scheduler-admit]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L1472-L1557
[nixl-plan]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py#L484-L504
[nixl-read]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py#L640-L654
[worker-kv]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/worker/gpu/kv_connector.py#L136-L153
[scheduler-resume]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/v1/core/sched/scheduler.py#L3593-L3629
[completion-generate]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/entrypoints/openai/completion/serving.py#L255-L288
[completion-stream]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/entrypoints/openai/completion/serving.py#L504-L553
[nixl-local-hit]: https://github.com/vllm-project/vllm/blob/a16bfee11b92f1809628a8bd1f4bd70cdcb9b576/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py#L135-L179
