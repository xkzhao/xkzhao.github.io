---
layout: post
title: "How Lucene Merges Segments While You Keep Searching"
date: 2026-11-21 14:50 -0500
description: "Follow one delete through a Lucene 10.6.0 segment merge. How TieredMergePolicy picks the inputs, how doc IDs get remapped, how a delete that lands mid-merge still reaches the output, and why disk usage rises before it falls."
categories: [Search, Database]
tags: [lucene, search-engine, java, indexing]
---

Part 3 of the Lucene internals series. [Part 1 follows documents into immutable segments]({% post_url 2026-06-14-deep-dive-into-apache-lucene-core %}). [Part 2 follows a query across those segments]({% post_url 2026-08-30-deep-dive-into-apache-lucene-search %}).

Part 2 left the two-million-review index sitting in twelve segments, with `quick OR dog` visiting every one of them. This part is about where those twelve came from and what happens when Lucene turns some of them into one.

Here is the awkward moment. A merge has been running for four seconds. It has already copied review 103 into the new segment. Now a delete for review 103 arrives. Seconds later the merge finishes and the new segment goes live. Does review 103 come back?

It does not, and the reason is not luck. Lucene snapshots which documents were live when the merge started, keeps a map from every input doc ID to its slot in the output, and replays anything that changed in between before the new segment is ever visible. This article follows that one delete through the machinery, with every claim checked against the Lucene 10.6.0 source.

> Every line number in this article refers to the `releases/lucene/10.6.0` tag. Two of the figures and the whole results table come from running code, not from reading it.
{: .prompt-info }

## Why there is anything to merge

A flush turns the in-memory buffer into a segment on disk. One flush adds at least one segment, one per indexing thread's buffer, so a busy writer produces them in bursts. Nothing rewrites an existing segment, because segments are immutable.

That immutability is also why a delete does not free anything. `IndexWriter` records the doc ID in a live-docs bitset beside the segment and leaves the bytes alone. The document stops matching queries immediately. Its postings, stored fields, norms and doc values stay on disk until something copies the segment's survivors somewhere else.

So two costs grow on their own. Query work grows with the segment count, because a term query does one term-dictionary seek and builds one scorer per segment ([`TermStates.java:103`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TermStates.java#L103), [`TermQuery.java:141`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/search/TermQuery.java#L141)). Dead bytes grow with the delete rate. A merge is the one operation that pays both down.

<svg viewBox="0 0 720 250" role="img" aria-label="Flushes add segments over time while deletes accumulate inside them, so a query must seek the term dictionary once per segment" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">Both costs grow without anyone doing anything wrong</text>
  <text x="0" y="40" font-size="11" fill="currentColor" opacity="0.75">flush</text>
  <text x="112" y="40" font-size="11" fill="currentColor" opacity="0.75">flush</text>
  <text x="224" y="40" font-size="11" fill="currentColor" opacity="0.75">flush</text>
  <text x="336" y="40" font-size="11" fill="currentColor" opacity="0.75">flush</text>
  <text x="448" y="40" font-size="11" fill="currentColor" opacity="0.75">flush</text>
  <rect x="0" y="48" width="86" height="34" rx="3" fill="#3b82f6"/><text x="43" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_0</text>
  <rect x="112" y="48" width="86" height="34" rx="3" fill="#3b82f6"/><text x="155" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_1</text>
  <rect x="224" y="48" width="86" height="34" rx="3" fill="#3b82f6"/><text x="267" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_2</text>
  <rect x="336" y="48" width="86" height="34" rx="3" fill="#3b82f6"/><text x="379" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_3</text>
  <rect x="448" y="48" width="86" height="34" rx="3" fill="#3b82f6"/><text x="491" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_4</text>
  <text x="556" y="70" font-size="12" fill="currentColor" opacity="0.8">and so on</text>
  <rect x="58" y="48" width="14" height="34" fill="#ef4444" opacity="0.85"/>
  <rect x="126" y="48" width="14" height="34" fill="#ef4444" opacity="0.85"/>
  <rect x="168" y="48" width="14" height="34" fill="#ef4444" opacity="0.85"/>
  <rect x="252" y="48" width="14" height="34" fill="#ef4444" opacity="0.85"/>
  <rect x="392" y="48" width="14" height="34" fill="#ef4444" opacity="0.85"/>
  <rect x="0" y="94" width="12" height="12" rx="2" fill="#3b82f6"/><text x="18" y="104" font-size="11" fill="currentColor">live documents</text>
  <rect x="140" y="94" width="12" height="12" rx="2" fill="#ef4444" opacity="0.85"/><text x="158" y="104" font-size="11" fill="currentColor">deleted, still on disk</text>
  <line x1="0" y1="124" x2="720" y2="124" stroke="currentColor" stroke-width="1" opacity="0.25"/>
  <text x="0" y="150" font-size="12" font-weight="600" fill="currentColor">What one term query now costs</text>
  <circle cx="60" cy="186" r="20" fill="none" stroke="currentColor" stroke-width="1.3"/>
  <text x="60" y="190" font-size="10.5" fill="currentColor" text-anchor="middle">query</text>
  <line x1="82" y1="186" x2="150" y2="166" stroke="currentColor" stroke-width="1.1" opacity="0.7"/>
  <line x1="82" y1="186" x2="150" y2="179" stroke="currentColor" stroke-width="1.1" opacity="0.7"/>
  <line x1="82" y1="186" x2="150" y2="193" stroke="currentColor" stroke-width="1.1" opacity="0.7"/>
  <line x1="82" y1="186" x2="150" y2="206" stroke="currentColor" stroke-width="1.1" opacity="0.7"/>
  <text x="158" y="170" font-size="11" fill="currentColor">seek the term dictionary in _0, build a scorer</text>
  <text x="158" y="183" font-size="11" fill="currentColor">seek the term dictionary in _1, build a scorer</text>
  <text x="158" y="196" font-size="11" fill="currentColor">seek the term dictionary in _2, build a scorer</text>
  <text x="158" y="209" font-size="11" fill="currentColor" opacity="0.8">once per segment, every query</text>
  <text x="0" y="238" font-size="12" fill="currentColor" opacity="0.85">A merge is the only thing that reduces either number.</text>
</svg>

## Which segments Lucene picks

`TieredMergePolicy` is the default. Three ideas carry most of its behaviour, and the third one is the one people are usually missing.

**It ranks segments by size, not by position.** `getSortedBySegmentSize` sorts every eligible segment by byte size descending, ties broken by segment name ([`TieredMergePolicy.java:341`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L341)). The size it sorts on is not the size on disk. `MergePolicy.size` prorates the byte size by the fraction of deleted documents ([`MergePolicy.java:761`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergePolicy.java#L761)), so a segment that is 40% deleted counts as 60% of its bytes. Candidates are then taken as a run through that size-sorted list, which is why a merge set can be scattered all over the index. Non-adjacency is not a feature bolted on, it falls out of ranking by size.

**It scores every window and keeps the best.** The outer loop tries every start index in the sorted list ([`:574`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L574)) and scores each candidate with three factors ([`:739-765`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L739)). Lower is better.

~~~java
double mergeScore = skew;                                   // largest input / total, floored
mergeScore *= Math.pow((double) totAfterMergeBytes, 0.05);   // "gently favor smaller merges"
mergeScore *= Math.pow(nonDelRatio, 2);                      // "strongly favor merges that reclaim deletes"
~~~

The exponents are the whole story. Skew is the base, deletes are squared, and output size is raised to 0.05, which is almost flat. The code comments say so in as many words, "Strongly favor merges with less skew", "Gently favor smaller merges over bigger ones", "Strongly favor merges that reclaim deletes". A candidate that is half deletes gets its score multiplied by 0.25, which is a far bigger lever than any plausible size difference.

<svg viewBox="0 0 720 300" role="img" aria-label="Segments sorted by live bytes descending, with a sliding candidate window scored by skew, size and deletes reclaimed" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">Sorted by live bytes, then every window is scored</text>
  <text x="0" y="38" font-size="11" fill="currentColor" opacity="0.75">eligible segments, largest live size first</text>
  <rect x="0" y="48" width="96" height="46" rx="3" fill="#3b82f6"/><text x="48" y="70" font-size="11" fill="#ffffff" text-anchor="middle">_7</text><text x="48" y="85" font-size="9.5" fill="#ffffff" text-anchor="middle" opacity="0.9">2.1 GB</text>
  <rect x="104" y="54" width="92" height="40" rx="3" fill="#3b82f6"/><text x="150" y="74" font-size="11" fill="#ffffff" text-anchor="middle">_2</text><text x="150" y="88" font-size="9.5" fill="#ffffff" text-anchor="middle" opacity="0.9">1.8 GB</text>
  <rect x="204" y="62" width="84" height="32" rx="3" fill="#3b82f6"/><text x="246" y="79" font-size="11" fill="#ffffff" text-anchor="middle">_b</text><text x="246" y="91" font-size="9.5" fill="#ffffff" text-anchor="middle" opacity="0.9">1.1 GB</text>
  <rect x="296" y="66" width="80" height="28" rx="3" fill="#6366f1"/><text x="336" y="81" font-size="11" fill="#ffffff" text-anchor="middle">_4</text><text x="336" y="91" font-size="9" fill="#ffffff" text-anchor="middle" opacity="0.9">0.9 GB</text>
  <rect x="384" y="68" width="76" height="26" rx="3" fill="#6366f1"/><text x="422" y="82" font-size="11" fill="#ffffff" text-anchor="middle">_9</text><text x="422" y="91" font-size="9" fill="#ffffff" text-anchor="middle" opacity="0.9">0.8 GB</text>
  <rect x="468" y="70" width="74" height="24" rx="3" fill="#6366f1"/><text x="505" y="83" font-size="11" fill="#ffffff" text-anchor="middle">_1</text><text x="505" y="91" font-size="9" fill="#ffffff" text-anchor="middle" opacity="0.9">0.8 GB</text>
  <rect x="550" y="78" width="58" height="16" rx="2" fill="#3b82f6" opacity="0.55"/><text x="579" y="90" font-size="9" fill="#ffffff" text-anchor="middle">_c</text>
  <rect x="616" y="82" width="46" height="12" rx="2" fill="#3b82f6" opacity="0.45"/>
  <rect x="670" y="85" width="34" height="9" rx="2" fill="#3b82f6" opacity="0.4"/>
  <rect x="292" y="42" width="254" height="60" rx="5" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="419" y="118" font-size="11" fill="#f59e0b" text-anchor="middle" font-weight="600">best-scoring window</text>
  <text x="0" y="146" font-size="11" fill="currentColor" opacity="0.75">index order of the same segments, for comparison</text>
  <rect x="0" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="28" y="167" font-size="9" fill="currentColor" text-anchor="middle">_0</text>
  <rect x="62" y="156" width="56" height="14" rx="2" fill="#6366f1"/><text x="90" y="167" font-size="9" fill="#ffffff" text-anchor="middle">_1</text>
  <rect x="124" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="152" y="167" font-size="9" fill="currentColor" text-anchor="middle">_2</text>
  <rect x="186" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="214" y="167" font-size="9" fill="currentColor" text-anchor="middle">_3</text>
  <rect x="248" y="156" width="56" height="14" rx="2" fill="#6366f1"/><text x="276" y="167" font-size="9" fill="#ffffff" text-anchor="middle">_4</text>
  <rect x="310" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="338" y="167" font-size="9" fill="currentColor" text-anchor="middle">...</text>
  <rect x="372" y="156" width="56" height="14" rx="2" fill="#6366f1"/><text x="400" y="167" font-size="9" fill="#ffffff" text-anchor="middle">_9</text>
  <rect x="434" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="462" y="167" font-size="9" fill="currentColor" text-anchor="middle">_a</text>
  <rect x="496" y="156" width="56" height="14" rx="2" fill="currentColor" opacity="0.18"/><text x="524" y="167" font-size="9" fill="currentColor" text-anchor="middle">_b</text>
  <text x="560" y="167" font-size="11" fill="currentColor" opacity="0.8">not adjacent in the index</text>
  <line x1="0" y1="188" x2="720" y2="188" stroke="currentColor" stroke-width="1" opacity="0.25"/>
  <text x="0" y="212" font-size="12" font-weight="600" fill="currentColor">The three score factors, drawn to their real weight</text>
  <text x="0" y="234" font-size="11" fill="currentColor">skew, largest input over total</text>
  <rect x="300" y="224" width="300" height="11" rx="2" fill="#10b981"/>
  <text x="610" y="234" font-size="10.5" fill="currentColor" opacity="0.8">base of the score</text>
  <text x="0" y="256" font-size="11" fill="currentColor">deletes reclaimed, squared</text>
  <rect x="300" y="246" width="300" height="11" rx="2" fill="#10b981"/>
  <text x="610" y="256" font-size="10.5" fill="currentColor" opacity="0.8">x0.25 at 50% deletes</text>
  <text x="0" y="278" font-size="11" fill="currentColor">output size, exponent 0.05</text>
  <rect x="300" y="268" width="26" height="11" rx="2" fill="#10b981"/>
  <text x="336" y="278" font-size="10.5" fill="currentColor" opacity="0.8">almost flat</text>
</svg>

**It refuses merges that barely grow.** A natural merge whose output would not be at least 50% larger than its biggest input is skipped outright ([`:623-636`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L623)). The comment in the source names the failure it prevents, "pathological O(N^2) merging where merges keep rewriting again and again the biggest input segment into a segment that is barely bigger". This is the rule that stops each new 20 MB flush from being folded into your 4 GB segment. The one exception is when that big segment has more than `deletesPctAllowed` deletes to reclaim, in which case rewriting it is the point.

<svg viewBox="0 0 720 200" role="img" aria-label="A merge of a 4 GB segment with a 20 MB segment is rejected because the output is not 50 percent larger, unless the big segment is full of deletes" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">The 50% growth rule, and its one exception</text>
  <text x="0" y="40" font-size="11" fill="currentColor" opacity="0.8">4.00 GB segment, 2% deleted, plus a fresh 20 MB flush</text>
  <rect x="0" y="50" width="400" height="26" rx="3" fill="#3b82f6"/><text x="200" y="68" font-size="11" fill="#ffffff" text-anchor="middle">4.00 GB</text>
  <rect x="404" y="50" width="6" height="26" rx="1" fill="#6366f1"/>
  <text x="420" y="68" font-size="11" fill="currentColor">output would be 4.02 GB</text>
  <text x="0" y="96" font-size="11.5" fill="#ef4444">Rejected. 4.02 is not 1.5 x 4.00, so the merge would rewrite 4 GB to gain 20 MB.</text>
  <line x1="0" y1="112" x2="720" y2="112" stroke="currentColor" stroke-width="1" opacity="0.2"/>
  <text x="0" y="136" font-size="11" fill="currentColor" opacity="0.8">same 4.00 GB segment, now 35% deleted</text>
  <rect x="0" y="146" width="260" height="26" rx="3" fill="#3b82f6"/>
  <rect x="260" y="146" width="140" height="26" rx="3" fill="#ef4444" opacity="0.85"/>
  <text x="130" y="164" font-size="11" fill="#ffffff" text-anchor="middle">2.60 GB live</text>
  <text x="330" y="164" font-size="11" fill="#ffffff" text-anchor="middle">1.40 GB dead</text>
  <text x="420" y="164" font-size="11" fill="currentColor">output would be 2.62 GB</text>
  <text x="0" y="192" font-size="11.5" fill="#10b981">Allowed. The biggest input is over deletesPctAllowed, so reclaiming its dead bytes is the point.</text>
</svg>

Those three ideas sit on top of a budget. `TieredMergePolicy` walks geometrically growing tiers, granting `segmentsPerTier` segments at each level and multiplying the level size by the merge factor as it goes ([`:441-460`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L441)). It merges when the index is over that segment budget, or when deleted documents exceed `deletesPctAllowed` of `maxDoc` ([`:562-566`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L562)).

The defaults matter here, and two of them moved recently enough that older write-ups have them wrong.

| Setting | 10.6.0 default | Note |
| --- | ---: | --- |
| `segmentsPerTier` | 8.0 | Was 10 before 10.3.0 |
| `floorSegmentMB` | 16 | Was 2 before 10.2.0, an 8x jump |
| `maxMergeAtOnce` | 10 | Effective merge factor is `min(10, 8)` = 8, so this knob rarely binds |
| `maxMergedSegmentMB` | 5120 | 5 GB ceiling on natural merges |
| `deletesPctAllowed` | 20.0 | Floor of the valid range was relaxed from 5 in 10.3.0 |
| `forceMergeDeletesPctAllowed` | 10.0 | Only `forceMergeDeletes` uses it |
| `targetSearchConcurrency` | 1 | Off by default, caps docs per segment when raised |

`floorSegmentMB` does double duty. It is the starting level size for the budget, which keeps a tail of tiny segments from inflating the allowed count, and it is the floor applied inside the skew term, so segments under 16 MB all look the same size and a merge of many tiny ones does not look lopsided. Below the floor Lucene will also merge more than the merge factor at once ([`:583-587`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java#L583)), so "at most 8 inputs" is not true at the small end.

## One review gets a new doc ID

Zoom all the way in. Five reviews, two segments, one of them already deleted.

Segment `_0` holds reviews 101, 102 and 103 at local doc IDs 0, 1 and 2. Review 102 was deleted before the merge started. Segment `_1` holds reviews 104 and 105. The application knows these documents by the indexed field `id`. Lucene knows them by doc ID, which is nothing more than a slot number inside one specific segment. A merge writes new slots, so every surviving document gets a new doc ID.

The order the inputs are concatenated in decides who gets the low slots, and it is worth being exact about it, because it surprised me. For an index with no index sort, the output is the inputs concatenated in the order the merge policy listed them, with each input's deleted docs squeezed out. Nothing in `IndexWriter` or `MergeState` reorders that list. And `TieredMergePolicy` lists them largest-first by *live* bytes, so the delete on review 102 shrinks `_0` below `_1` and `_1` goes first. Take the delete away and `_0` leads instead. I verified both directions by running it.

<svg viewBox="0 0 720 320" role="img" aria-label="Two input segments merge into one output segment; the deleted document is dropped and the remaining documents get new doc IDs, recorded in a doc map" style="width:100%;height:auto;max-width:720px;">
  <defs><marker id="lm-ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">The output is a concatenation with the dead slots squeezed out</text>
  <text x="0" y="42" font-size="11" fill="currentColor" opacity="0.75">input _1, 1,977 bytes with nothing deleted, so it sorts first</text>
  <rect x="0" y="52" width="104" height="30" rx="3" fill="#3b82f6"/><text x="52" y="72" font-size="11" fill="#ffffff" text-anchor="middle">0 · 104</text>
  <rect x="110" y="52" width="104" height="30" rx="3" fill="#3b82f6"/><text x="162" y="72" font-size="11" fill="#ffffff" text-anchor="middle">1 · 105</text>
  <text x="0" y="106" font-size="11" fill="currentColor" opacity="0.75">input _0, 2,044 bytes but one third deleted, so 1,362 live bytes</text>
  <rect x="0" y="116" width="104" height="30" rx="3" fill="#6366f1"/><text x="52" y="136" font-size="11" fill="#ffffff" text-anchor="middle">0 · 101</text>
  <rect x="110" y="116" width="104" height="30" rx="3" fill="none" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4 3"/><text x="162" y="131" font-size="11" fill="#ef4444" text-anchor="middle">1 · 102</text><text x="162" y="142" font-size="9" fill="#ef4444" text-anchor="middle">deleted</text>
  <rect x="220" y="116" width="104" height="30" rx="3" fill="#f59e0b"/><text x="272" y="136" font-size="11" fill="#ffffff" text-anchor="middle">2 · 103</text>
  <line x1="360" y1="100" x2="420" y2="100" stroke="currentColor" stroke-width="1.4" marker-end="url(#lm-ar)"/>
  <text x="390" y="92" font-size="10.5" fill="currentColor" text-anchor="middle" opacity="0.8">merge</text>
  <text x="440" y="42" font-size="11" fill="currentColor" opacity="0.75">output _2</text>
  <rect x="440" y="52" width="130" height="26" rx="3" fill="#10b981"/><text x="505" y="69" font-size="11" fill="#ffffff" text-anchor="middle">0 · 104</text>
  <rect x="440" y="82" width="130" height="26" rx="3" fill="#10b981"/><text x="505" y="99" font-size="11" fill="#ffffff" text-anchor="middle">1 · 105</text>
  <rect x="440" y="112" width="130" height="26" rx="3" fill="#10b981"/><text x="505" y="129" font-size="11" fill="#ffffff" text-anchor="middle">2 · 101</text>
  <rect x="440" y="142" width="130" height="26" rx="3" fill="#f59e0b"/><text x="505" y="159" font-size="11" fill="#ffffff" text-anchor="middle">3 · 103</text>
  <text x="582" y="160" font-size="10.5" fill="#f59e0b">the one we follow</text>
  <line x1="0" y1="192" x2="720" y2="192" stroke="currentColor" stroke-width="1" opacity="0.25"/>
  <text x="0" y="216" font-size="12" font-weight="600" fill="currentColor">The doc maps MergeState hands the codec, old ID to new ID</text>
  <text x="0" y="240" font-size="11" fill="currentColor" opacity="0.8">docMaps[0], for _1</text>
  <text x="200" y="240" font-size="11.5" fill="currentColor" font-family="monospace">0 -&gt; 0     1 -&gt; 1</text>
  <text x="0" y="262" font-size="11" fill="currentColor" opacity="0.8">docMaps[1], for _0</text>
  <text x="200" y="262" font-size="11.5" fill="currentColor" font-family="monospace">0 -&gt; 2     1 -&gt; -1     2 -&gt; 3</text>
  <text x="440" y="262" font-size="10.5" fill="#ef4444">-1 means "do not write this one"</text>
  <text x="0" y="292" font-size="12" fill="currentColor" opacity="0.85">The codec never seeks to a slot. It pulls documents through a DocIDMerger that consults these</text>
  <text x="0" y="310" font-size="12" fill="currentColor" opacity="0.85">maps, skips every -1, and appends whatever it is handed.</text>
</svg>

That last point is the part worth internalising. `MergeState.docMaps` is documented as mapping "document IDs from old segments to document IDs in the new segment" ([`MergeState.java:46`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java#L46)), and for an unsorted index each map is a doc base plus a packed table of "how many live docs came before me", returning -1 for anything deleted ([`:190-207`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java#L190)). The codecs do not read those maps directly. They wrap each input in a `DocIDMerger`, which asks the map for a mapped ID, skips the -1s, and yields documents in output order ([`DocIDMerger.java:58-69`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/DocIDMerger.java#L58)). `SegmentMerger` then drives every per-segment structure off the same merge state, stored fields first, then norms, postings, doc values, points, vectors, term vectors and field infos ([`SegmentMerger.java:117-179`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/SegmentMerger.java#L117)).

If the index has an index sort, the concatenation story goes away. `MergeState` routes to `MultiSorter`, which runs a real merge sort across all the inputs so their documents interleave by sort key ([`MergeState.java:214-240`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java#L214)).

## How the late delete gets in

Now the awkward moment from the opening. The merge has copied review 103 into `_2` at slot 3. The delete arrives. Three facts make this safe.

**The merge takes a snapshot.** Before any reader is opened, `mergeInit` waits for every buffered delete packet that applies to these segments to finish resolving ([`IndexWriter.java:5375`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L5375)). Then each merge reader records the input's hard live docs as they stood at that instant ([`ReadersAndUpdates.java:1166`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ReadersAndUpdates.java#L1166)). That is what gives the merge a clean "before".

**Live docs are copy-on-write.** A delete arriving later cannot mutate the snapshot. `PendingDeletes` drops its writeable bitset once the live docs are handed out, so the next delete has to clone first ([`PendingDeletes.java:108`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/PendingDeletes.java#L108)). That is why Lucene can test "did anything change?" with a single reference comparison.

**The carry-over happens under the writer lock, before publication.** `commitMerge` is `synchronized` and calls `commitMergedDeletesAndUpdates` before `segmentInfos.applyMergeChanges` splices the new segment in ([`IndexWriter.java:5049`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L5049), install at [`:5102`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L5102)). For each input it compares the snapshot against the current hard live docs and, for every document deleted in between, marks the mapped output slot deleted ([`:4947-5004`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L4947)).

~~~java
if (currentHardLiveDocs != prevHardLiveDocs) {        // copy-on-write, so identity is enough
  for (int j = 0; j < maxDoc; j++) {
    if (prevHardLiveDocs.get(j) == false) {
      assert currentHardLiveDocs.get(j) == false;     // a deleted doc can never come back
    } else if (carryOverDelete.test(j)) {
      mergedReadersAndUpdates.delete(segDocMap.get(j));
    }
  }
}
~~~

The same pass remaps the doc-values updates that landed during the merge and survived it ([`:4576-4676`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L4576)). And if every document in the output ends up deleted this way, `commitMerge` throws the whole segment away instead of installing it ([`:5059`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L5059)).

<svg viewBox="0 0 720 300" role="img" aria-label="Timeline of a merge showing the live-docs snapshot, the delete arriving mid-merge, the warmer running before publication, and the carry-over under the writer lock" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">Where the late delete slots in</text>
  <line x1="92" y1="46" x2="700" y2="46" stroke="currentColor" stroke-width="1" opacity="0.3"/>
  <text x="0" y="50" font-size="11" font-weight="600" fill="currentColor">merge thread</text>
  <rect x="92" y="34" width="86" height="24" rx="3" fill="#6366f1"/><text x="135" y="50" font-size="10" fill="#ffffff" text-anchor="middle">pull readers</text>
  <rect x="182" y="34" width="250" height="24" rx="3" fill="#3b82f6"/><text x="307" y="50" font-size="10" fill="#ffffff" text-anchor="middle">write _2, copying survivors</text>
  <rect x="436" y="34" width="84" height="24" rx="3" fill="#6366f1" opacity="0.8"/><text x="478" y="50" font-size="10" fill="#ffffff" text-anchor="middle">warmer</text>
  <rect x="524" y="34" width="110" height="24" rx="3" fill="#10b981"/><text x="579" y="50" font-size="10" fill="#ffffff" text-anchor="middle">commitMerge</text>
  <line x1="92" y1="104" x2="700" y2="104" stroke="currentColor" stroke-width="1" opacity="0.3"/>
  <text x="0" y="108" font-size="11" font-weight="600" fill="currentColor">app thread</text>
  <circle cx="330" cy="104" r="7" fill="#ef4444"/>
  <text x="330" y="86" font-size="10.5" fill="#ef4444" text-anchor="middle">delete id=103</text>
  <text x="330" y="128" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.8">lands on _0, slot 2</text>
  <line x1="135" y1="60" x2="135" y2="160" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>
  <line x1="579" y1="60" x2="579" y2="160" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>
  <text x="92" y="176" font-size="10.5" fill="currentColor">snapshot taken: _0 live = {0, 2}</text>
  <text x="524" y="176" font-size="10.5" fill="currentColor">_0 live now = {0}</text>
  <text x="92" y="200" font-size="11.5" fill="currentColor">The two differ by doc 2, and docMaps[1].get(2) = 3,</text>
  <text x="92" y="218" font-size="11.5" fill="currentColor">so slot 3 of _2 is marked deleted before _2 is installed.</text>
  <line x1="0" y1="238" x2="720" y2="238" stroke="currentColor" stroke-width="1" opacity="0.25"/>
  <text x="0" y="260" font-size="11.5" fill="#f59e0b">Note the order. The merged-segment warmer runs before commitMerge, so a warmed reader sees</text>
  <text x="0" y="278" font-size="11.5" fill="#f59e0b">the output without the carried-over delete. That window is exactly what the example below pauses in.</text>
  <text x="0" y="296" font-size="11" fill="currentColor" opacity="0.7">IndexWriter.java:5908 warms, IndexWriter.java:5922 commits.</text>
</svg>

That timing is why the output has four physical slots and three live reviews. Review 103's bytes were already written, so they stay. `maxDoc` is fixed at merge time from the inputs' live counts ([`MergeState.java:170`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java#L170)) and a later delete never shrinks it. Search skips the slot, `MergePolicy.size` discounts the segment so it becomes a merge candidate sooner, and the next merge is what finally reclaims the space.

## Which reader sees the result

A merge changes what the index looks like, but not for everyone at once. Five views of the same instant disagree, and each disagreement has a cause.

<svg viewBox="0 0 720 280" role="img" aria-label="Five views of the index during and after a merge, showing which ones find review 103 and which do not" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">Five views, two answers</text>
  <line x1="232" y1="34" x2="232" y2="246" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="232" y="266" font-size="10.5" fill="#f59e0b" text-anchor="middle">commitMerge installs _2</text>
  <line x1="560" y1="34" x2="560" y2="246" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="560" y="266" font-size="10.5" fill="#10b981" text-anchor="middle">writer.commit writes segments_N</text>
  <rect x="0" y="36" width="224" height="26" rx="3" fill="#6366f1"/><text x="112" y="53" font-size="10.5" fill="#ffffff" text-anchor="middle">merged segment, inside the warmer</text>
  <text x="240" y="53" font-size="11" fill="currentColor">finds 103 at slot 3, 4 live / 4 slots</text>
  <rect x="0" y="76" width="700" height="26" rx="3" fill="#3b82f6" opacity="0.85"/><text x="120" y="93" font-size="10.5" fill="#ffffff" text-anchor="middle">the reader opened before the merge, still open</text>
  <text x="400" y="93" font-size="10.5" fill="#ffffff">finds 103 at slot 2 of _0, 4 live / 5 slots</text>
  <rect x="240" y="116" width="460" height="26" rx="3" fill="#10b981"/><text x="470" y="133" font-size="10.5" fill="#ffffff" text-anchor="middle">reader reopened from the writer, 103 absent, 3 live / 4 slots</text>
  <rect x="240" y="156" width="312" height="26" rx="3" fill="#3b82f6" opacity="0.6"/><text x="396" y="173" font-size="10.5" fill="#ffffff" text-anchor="middle">directory reader, still on the old commit</text>
  <text x="560" y="173" font-size="10.5" fill="currentColor">finds 103</text>
  <rect x="568" y="196" width="132" height="26" rx="3" fill="#10b981"/><text x="634" y="213" font-size="10.5" fill="#ffffff" text-anchor="middle">103 absent</text>
  <text x="240" y="213" font-size="10.5" fill="currentColor">directory reader opened after the commit</text>
  <text x="0" y="240" font-size="11" fill="currentColor" opacity="0.8">Nothing in flight is interrupted. Every view is internally consistent.</text>
</svg>

The writer installs the new segment in its in-memory `SegmentInfos` and checkpoints without writing a `segments_N` ([`IndexWriter.java:2927`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L2927)). A near-real-time reader opened or reopened from that writer reads straight off it, so it sees the merge with no durable commit ([`:609`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L609)). An NRT reader that is *already* open sees nothing new until you reopen it. A reader opened from the directory resolves the highest `segments_N` it can find ([`StandardDirectoryReader.java:83`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/StandardDirectoryReader.java#L83)), so it stays on the last commit.

I checked all five against Lucene 10.6.0 rather than reasoning about them. The example pauses a forced merge inside its merged-segment warmer, which is the window after the output is written and before `commitMerge` runs, applies the delete there, lets the merge finish, and then runs a term query for `id=103` against each view. Every row below is an assertion in that program, so the program fails loudly if Lucene ever disagrees.

| View queried | Review 103 | Live docs / physical slots |
| --- | --- | ---: |
| New segment, inside the warmer, before install | Found at doc ID 3 | 4 / 4 |
| Original reader, still open | Found at doc ID 2 | 4 / 5 |
| Reader reopened from the writer | Absent | 3 / 4 |
| Directory reader opened before the next commit | Found at doc ID 2 | 4 / 5 |
| Directory reader opened after the next commit | Absent | 3 / 4 |

Run the [Java example]({{ '/assets/posts/lucene-segment-merges/LateDelete.java' | relative_url }}) with JDK 21 or later and the [Lucene 10.6.0 core JAR](https://repo.maven.apache.org/maven2/org/apache/lucene/lucene-core/10.6.0/lucene-core-10.6.0.jar) in the same directory.

~~~shell
java -cp lucene-core-10.6.0.jar LateDelete.java
~~~

One caveat on reading this as a model of hard deletes in general. `carryOverHardDeletes` only handles hard deletes. A soft delete is a doc-values write, so it travels on the doc-values carry-over path instead, and the merge wraps its readers to keep soft-deleted documents alive while still dropping hard-deleted ones ([`IndexWriter.java:5622-5661`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L5622)).

## Why disk usage rises before it falls

Lucene writes the output while the inputs are still there, so peak usage is roughly inputs plus output. The interesting question is what happens after the swap, and the answer is reference counting, not garbage collection.

<svg viewBox="0 0 720 230" role="img" aria-label="Disk usage before, during and after a merge, rising while inputs and output coexist and falling once the inputs are released" style="width:100%;height:auto;max-width:720px;">
  <text x="0" y="16" font-size="13" font-weight="600" fill="currentColor">Disk used by the segments in one merge</text>
  <rect x="0" y="30" width="12" height="12" rx="2" fill="#3b82f6"/><text x="18" y="40" font-size="11" fill="currentColor">input segments</text>
  <rect x="150" y="30" width="12" height="12" rx="2" fill="#10b981"/><text x="168" y="40" font-size="11" fill="currentColor">output segment</text>
  <text x="0" y="72" font-size="11.5" fill="currentColor">before the merge</text>
  <rect x="150" y="60" width="278" height="16" rx="2" fill="#3b82f6"/>
  <text x="440" y="73" font-size="11.5" fill="currentColor" font-variant-numeric="tabular-nums">2.0 GiB</text>
  <text x="0" y="108" font-size="11.5" fill="currentColor">inputs and output coexist</text>
  <rect x="150" y="96" width="278" height="16" rx="2" fill="#3b82f6"/>
  <rect x="428" y="96" width="222" height="16" rx="2" fill="#10b981"/>
  <text x="662" y="109" font-size="11.5" fill="#f59e0b" font-variant-numeric="tabular-nums" font-weight="600">3.6 GiB</text>
  <text x="0" y="144" font-size="11.5" fill="currentColor">inputs released</text>
  <rect x="150" y="132" width="222" height="16" rx="2" fill="#10b981"/>
  <text x="384" y="145" font-size="11.5" fill="currentColor" font-variant-numeric="tabular-nums">1.6 GiB</text>
  <line x1="650" y1="52" x2="650" y2="156" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="0" y="178" font-size="11" fill="currentColor" opacity="0.75">Illustrative sizes for one merge. Other segments and temporary files add to the real total.</text>
  <text x="0" y="206" font-size="11.5" fill="currentColor">Capacity planning follows the middle bar, not the bottom one. A 5 GB max segment size means</text>
  <text x="0" y="224" font-size="11.5" fill="currentColor">one merge alone can need 5 GB of headroom you never see in steady state.</text>
</svg>

`IndexFileDeleter` keeps a reference count per filename and deletes a file the instant its count reaches zero ([`FileDeleter.java:77-130`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/util/FileDeleter.java#L77)). Three things hold references, and it is worth knowing which is which:

- **The writer's own in-memory `SegmentInfos`**, through the non-commit checkpoint ([`IndexFileDeleter.java:552-586`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexFileDeleter.java#L552)).
- **NRT readers**, which call `incRefDeleter` on the segments they opened ([`StandardDirectoryReader.java:149`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/StandardDirectoryReader.java#L149)).
- **Retained commits**, if you replace the default deletion policy. `KeepOnlyLastCommitDeletionPolicy` drops every earlier commit as soon as a new one lands, so by default this holds nothing.

A plain directory reader is the one that surprises people. It holds no reference count at all. Its files get unlinked out from under it, and it keeps working only because of POSIX delete-on-last-close semantics, which the `IndexWriter` javadoc calls out explicitly. So on POSIX the unlink succeeds immediately and the blocks come back when the last reader closes its descriptors. Space is reclaimed when readers close, not when some background sweep notices.

Windows works differently, and "the operating system releases the file" is the wrong model there. The delete itself fails while a handle is open, so `FSDirectory` parks the filename in a pending-deletes set and retries later ([`FSDirectory.java:358-372`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/store/FSDirectory.java#L358)). Until a retry succeeds the bytes are still on disk, hidden from `listAll` and rejected by `openInput`. On Windows, a long-lived reader really can keep disk usage at the peak.

## What a merge costs while traffic is live

Once a merge is installed, every reader opened after it visits one segment where there were several. Readers already open, and queries already running, keep the old segment count until they reopen. Meanwhile the merge itself was reading and rewriting index data on the same machine that was serving queries.

Four mechanisms decide how much that hurts, and three of their defaults are not what the folklore says.

**Merges run on background threads, and can stall the indexer.** `ConcurrentMergeScheduler` stalls the calling thread while `mergeSource.hasPendingMerges() && mergeThreadCount() >= maxMergeCount` ([`ConcurrentMergeScheduler.java:619`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java#L619)). The thread it stalls is your application thread inside `addDocument` or `commit`. Merge threads are explicitly exempt, since stalling one would deadlock progress. Each wait is 250 ms and the condition is rechecked, so the total stall is unbounded. The trigger is `maxMergeCount`, not `maxThreadCount`.

**The defaults are derived from core count, with no disk probe.** `maxThreadCount = max(1, coreCount / 2)` and `maxMergeCount = maxThreadCount + 5` ([`:192`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java#L192)). The old SSD-versus-spinning-disk detection is gone. The class now assumes an SSD and leaves the conservative spinning-disk numbers as a manual opt-in. The class javadoc still advertises a `min(4, cores/2)` cap that the code does not implement, so do not trust that line.

**Auto IO throttle exists but is off.** `doAutoIOThrottle = false` by default ([`:112`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java#L112)), so every non-forced merge runs at `Double.POSITIVE_INFINITY` MB/s unless you call `enableAutoIOThrottle()`. What the scheduler does do unconditionally is pause the largest merges to zero when more than `maxThreadCount` merges over 50 MB are running ([`:373`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java#L373)), which is how small merges get to jump the queue. Note also that rate limiting, when enabled, limits merge *writes* only. Reads are untouched.

**Sequential read advice for merges is opt-in.** This one is easy to get wrong, and I had it wrong. A merge does get its own memory mapping, because every `MMapDirectory.openInput` call builds a fresh one, so the advice on a merge's mapping is independent of the search readers' mappings. And `MMapDirectory.ADVISE_BY_CONTEXT` does turn `Context.MERGE` into `ReadAdvice.SEQUENTIAL` ([`MMapDirectory.java:156-169`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/store/MMapDirectory.java#L156)). But `ADVISE_BY_CONTEXT` is not installed by default. The field is an empty-returning function, so every mapping falls back to `Constants.DEFAULT_READADVICE`, which is `ReadAdvice.NORMAL` ([`:171`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/store/MMapDirectory.java#L171)). Nothing in released core calls `setReadAdvice`. To get the behaviour you have to ask for it:

~~~java
var dir = new MMapDirectory(path);
dir.setReadAdvice(MMapDirectory.ADVISE_BY_CONTEXT);
~~~

Even then `madvise` is best effort. It is issued on platforms like Linux and macOS, and only when native access is permitted, which on a modern JDK means `--enable-native-access=org.apache.lucene.core`.

Two more costs the segment-count story hides. Some codecs parallelise *inside* one merge through `MergeScheduler.getIntraMergeExecutor`, but in released core only the HNSW vector writers use it, so this matters for vector indexes and not for a text-only one. And opening an NRT reader can itself run a merge on your calling thread, bounded by `maxFullFlushMergeWaitMillis`, which defaults to 500 ms ([`IndexWriterConfig.java:108`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriterConfig.java#L108)). A refresh is not free.

### How to see it in your own workload

Chart p99 query latency, indexing throughput, segment count and disk usage on one time axis. Mark when each merge starts and ends, and when the searcher refreshes. Hold the offered query rate, update rate and refresh interval steady, because changing any of them changes the merge schedule too.

Then read the three marks separately. Merge end is when the rewrite finished. Refresh is when queries start using the replacement. The old files can outlive both, for exactly the reference-count reasons above. An indexing slowdown and a latency spike from the same merge backlog will not line up, because one comes from `maybeStall` on the write path and the other from cache and IO contention on the read path.

## Reference

Everything above is from the `releases/lucene/10.6.0` tag.

| Topic | Source |
| --- | --- |
| Candidate selection, scoring, growth rule | [`TieredMergePolicy`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java) |
| Live-size prorating, `OneMerge`, merge readers | [`MergePolicy`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergePolicy.java) |
| Doc maps, index-sort routing, output `maxDoc` | [`MergeState`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java) |
| Writing the merged structures | [`SegmentMerger`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/SegmentMerger.java) |
| Warmer, `commitMerge`, late-delete carry-over | [`IndexWriter`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java) |
| Snapshotting live docs for a merge | [`ReadersAndUpdates`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ReadersAndUpdates.java) |
| Reference counting and file lifetime | [`IndexFileDeleter`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexFileDeleter.java) |
| NRT versus committed views | [`StandardDirectoryReader`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/StandardDirectoryReader.java) |
| Stalls, pausing, IO throttle, intra-merge executor | [`ConcurrentMergeScheduler`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java) |
| Read advice per mapping | [`MMapDirectory`](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/store/MMapDirectory.java) |

Also useful: the [Lucene 10.6.0 release notes](https://github.com/apache/lucene/releases/tag/releases%2Flucene%2F10.6.0) and Mike McCandless's [near-real-time benchmarks](https://benchmarks.mikemccandless.com/nrt.html).
