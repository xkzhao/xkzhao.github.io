---
layout: post
title: "LLM Inference: KV Cache, Prefill, and Decode"
date: 2026-09-23 22:27 -0400
categories: [AI, LLM Inference]
tags: [llm, inference, kv-cache, prefill, decode]
---

You send a prompt to an LLM endpoint. The server spends a moment working without returning text, then words begin to stream. The first word can take much longer than the gaps between later words. Increasing prompt length often changes that first wait; increasing answer length mostly changes how long the stream continues. Why does one request have two such different rhythms?

The answer lies in a dependency. Every prompt token is known when the request arrives. A generated token is unknown until the model chooses it, and the following token depends on that choice. Serving systems call the initial processing of known prompt tokens **prefill** and the repeated processing of newly chosen tokens **decode**. This distinction shapes latency, GPU use, memory traffic, and nearly every optimization later in this series.



## Prefill and Decode

A tokenizer converts input text to **token IDs**, not directly to vectors. The model's embedding layer turns those IDs into hidden states. In each attention layer, a position produces a query, key, and value (Q, K, V). A causal mask lets that position attend only to positions at or before itself.

The complete prompt is known on arrival. During **prefill**, the model processes its positions with parallel tensor operations while preserving that causal rule. The last prompt position supplies the logits used to choose the **first output token**. Prefill also writes K and V for the prompt to the KV cache. During ordinary **decode**, the chosen output token becomes the next model input; its query attends to prior K and V plus its own new K and V, the new K and V join the cache, and the model chooses another token. The next choice cannot be made until the previous one exists. Other requests can still share a GPU iteration through batching.

![Request timeline: prefill creates the first output token and prompt KV; later decode steps append KV and create subsequent tokens](/assets/img/posts/kv-prefill-decode-timeline.svg)

*Figure 1. Conceptual token path. For an output of O tokens, the ordinary cached path makes one prefill choice and O − 1 subsequent decode choices. Chunked prefill and speculative decoding can change the execution schedule.*

There is a useful cache-boundary detail here. Prefill caches the **prompt positions** and chooses output token 1. The next decode pass consumes token 1, computes its K/V, and chooses output token 2. Immediately after token 1 is emitted, its own K/V need not yet be in the cache. This distinction prevents the common off-by-one error in token and memory accounting.

The shape difference matters. A 2,048-token prompt offers large matrix operations over many known positions. A single sequence's decode step offers only one new position, but still runs the model layers and reads prior KV. Prefill therefore **often** has better arithmetic reuse; small-batch decode **often** has greater sensitivity to weight and KV bandwidth. Neither phase has a universal bottleneck: context length, batch size, kernels, GPU placement, and scheduling can change it. The [Transformer paper](https://arxiv.org/abs/1706.03762) establishes causal masking; the [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) explains why storing the resulting KV state becomes a serving constraint.

## What the cache saves, and what it does not

Without a cache, a straightforward decode implementation would process the prompt and all generated tokens again for each next-token choice. In a causal decoder with fixed model configuration and positions, adding a new token does not change earlier positions' K and V. Keeping those tensors avoids recomputing them. It does **not** remove the new query's attention over visible history: decode still reads past K and V. Efficient attention kernels need not materialize an entire attention-score matrix to do that work.

![Causal attention matrix and the growing key-value cache](/assets/img/posts/kv-causal-cache-growth.svg)

*Figure 2. Each row is a query position; filled cells are visible keys. Prefill computes the prompt's triangular attention region and stores its K/V columns. A later decode query reads those columns, then extends the cache by one position. The drawing shows logical positions, not physical memory layout.*

Think of KV as a **live per-sequence working set**. It is tied to token IDs, positions, model weights, and attention configuration. A request can release it when generation ends or is cancelled. A different request may reuse an identical eligible prefix if the runtime keeps those blocks and validates the match, but ordinary in-flight KV is not a generic application result cache. That distinction underlies [prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/).

If live KV is evicted, continuing that sequence efficiently requires recomputing it from the token prefix or restoring it from another memory tier; the model has not permanently learned the conversation. Prefix reuse is also subject to tenant isolation: [vLLM's security guidance](https://docs.vllm.ai/en/latest/usage/security/) describes a timing side channel when cache hits reveal another tenant's activity.

## Two clocks and one memory budget

**Time to first token (TTFT)** is the wait from a stated arrival boundary to the first output token. Client-observed TTFT includes network, admission queue, tokenization, prefill, first-token selection, and response transport. Server metrics may use a different start point. **Inter-token latency (ITL)** is each gap between successive emitted tokens; in a stream that bundles several tokens into one event, event gaps are only a proxy. A common per-request **time per output token (TPOT)** definition averages the gaps after the first token:

~~~text
TPOT = (last-token time − first-token time) / (output tokens − 1)
end-to-end latency = TTFT + sum of later token gaps
~~~

TPOT is undefined for a one-token output. [vLLM's current metric definitions](https://docs.vllm.ai/en/latest/design/metrics/) distinguish event-level ITL from per-request TPOT; record the runtime version and metric boundary when comparing systems. A hypothetical 200 ms TTFT followed by fifteen 40 ms gaps gives a 16-token response in 800 ms. Those inputs illustrate why a quick first token and smooth subsequent tokens deserve separate SLOs; they are not a performance target.

KV capacity has a similarly explicit budget. For a full-attention decoder with uniform layers and no sharing, the **logical payload** is:

~~~text
KV bytes = 2 × layers × KV heads × head dimension × bytes per element
           × cached positions per sequence × active sequences
~~~

The leading 2 counts K and V. [Meta's Llama 3.1 70B configuration](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py) specifies 80 layers, 64 query heads, eight KV heads, and width 8,192, giving head dimension 128. With BF16 KV at two bytes per element, that is 2 × 80 × 8 × 128 × 2 = **327,680 bytes = 320 KiB per cached token across the model**.

| Resident positions | Active sequences | Logical KV across model | Ideal share per GPU at TP8 |
| ---: | ---: | ---: | ---: |
| 1,024 | 1 | 320 MiB | 40 MiB |
| 8,192 | 1 | 2.5 GiB | 320 MiB |
| 8,192 | 16 | 40 GiB | 5 GiB |
| 32,768 | 16 | 160 GiB | 20 GiB |

The final column assumes perfectly even KV-head sharding over eight GPUs. A real runtime also needs weights, temporary workspaces, block metadata, and allocation headroom. This table cannot establish how many requests an 80 GB GPU will admit.

The formula describes logical payload, not physical allocation. Partly filled pages can waste space; prefix sharing can let several requests reference the same physical blocks; KV quantization changes bytes per element. Sliding-window or hybrid attention needs a different length model, and some parallel layouts replicate KV rather than split it evenly. Inspect the runtime's cache layout before translating this table into a GPU purchase or admission limit.

![Analytical KV payload rises with context length and concurrency](/assets/img/posts/kv-memory-scaling.svg)

*Figure 3. Analytical logical payload for the BF16 Llama 3.1 70B configuration. The dashed 40 GiB line is an illustrative KV budget across the model, not a measured capacity. Longer prompts consume cache before the first token; each subsequent decode step grows it again.*


## What this changes in a serving design

Prefill and decode are not two products or two model copies. They are phases of one autoregressive request, with different tensor shapes and reuse opportunities. A runtime may execute them on the same GPU worker, mix them in a batch, or place them on different workers. Each choice trades throughput, first-token latency, token gaps, and the cost of moving saved attention state.

For an interactive chat application, TTFT and tail ITL deserve separate SLOs: a fast first token cannot compensate for repeated pauses, and smooth streaming cannot fully compensate for a long initial wait. For offline batch generation, total tokens per second and cost may matter more than either streaming metric. Long-context retrieval workloads put more pressure on prefill and saved state. A high-concurrency API adds queues and competition between phases. The same Llama model can therefore have different bottlenecks under different traffic.
