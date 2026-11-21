---
layout: post
title: "How Lucene Merges Segments While You Keep Searching"
date: 2026-11-21 14:50 -0500
description: "Follow a delete through a Lucene 10.6.0 segment merge, from doc ID remapping to reader refresh, and see why disk usage can rise before it falls."
categories: [Search, Database]
tags: [lucene, search-engine, java, indexing]
---

Part 3 of the Lucene internals series. [Part 1 follows documents into immutable segments]({% post_url 2026-06-14-deep-dive-into-apache-lucene-core %}) and [Part 2 follows a query across those segments]({% post_url 2026-08-30-deep-dive-into-apache-lucene-search %}).

A search for `quick OR dog` is running against our two-million-review index. At the same time, Lucene is merging two segments. The merge has already copied review 103 into the new segment when a delete for that review arrives. When the output becomes searchable, will review 103 come back?

An open reader keeps the view it already had. A refreshed reader sees the replacement segment with review 103 marked deleted. Lucene keeps track of which documents are live in each view and maps old document IDs to their new locations.

## Which segments does Lucene merge?

Every flush can add a segment. Each additional segment gives a term query another place to look up terms and initialize its scorers. Deletes also leave bytes behind. Lucene marks the old document as deleted and can reclaim its space when it rewrites the segment.

`TieredMergePolicy` estimates how many segments the index can support across a range of sizes. When it needs a merge, it evaluates candidates using their estimated live byte sizes. It favors balanced inputs, smaller output, and candidates that reclaim more deleted data. It can pick segments that are far apart in the index. Repeatedly merging tiny new segments into the same large segment would rewrite that large segment over and over. [Lucene 10.6.0 policy source](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/TieredMergePolicy.java)

## One review gets a new doc ID

Zoom in on five reviews across two segments. Review 102 is already deleted when the merge starts. Review 103 is still live in input segment `_0` at local doc ID 2. The application identifies it by the indexed field `id=103`. The internal doc ID identifies a slot in a specific segment.

The merge in this example reads `_1` first, so its two reviews take the first two output slots. Review 101 takes the next slot. The merge skips the deleted review 102 and gives review 103 output doc ID 3. Each box shows the local doc ID followed by the review ID.

<link rel="stylesheet" href="{{ '/assets/posts/lucene-segment-merges/figures.css' | relative_url }}">
<script defer src="{{ '/assets/posts/lucene-segment-merges/merge.js' | relative_url }}"></script>

<figure class="lucene-figure lucene-walkthrough">
  <figcaption class="lucene-figure-title">A delete arrives after the merge copies review 103</figcaption>
  <div class="lucene-steps">
    <div class="lucene-step">
      <p class="lucene-step-title">1. Open a reader over the input segments</p>
      <div class="lucene-row">
        <span class="lucene-row-label">Segment _0</span>
        <span class="lucene-doc">0 · 101</span>
        <span class="lucene-doc lucene-deleted"><s>1 · 102</s><small>Deleted</small></span>
        <span class="lucene-doc lucene-target">2 · 103</span>
      </div>
      <div class="lucene-row">
        <span class="lucene-row-label">Segment _1</span>
        <span class="lucene-doc">0 · 104</span>
        <span class="lucene-doc">1 · 105</span>
      </div>
      <p>The open reader can find review 103 at doc ID 2 in segment _0.</p>
    </div>
    <div class="lucene-step">
      <p class="lucene-step-title">2. Write the replacement segment</p>
      <div class="lucene-row">
        <span class="lucene-row-label">Output _2</span>
        <span class="lucene-doc">0 · 104</span>
        <span class="lucene-doc">1 · 105</span>
        <span class="lucene-doc">2 · 101</span>
        <span class="lucene-doc lucene-target">3 · 103</span>
      </div>
      <p>The merge drops review 102. Its doc map records that _0:2 became _2:3.</p>
    </div>
    <div class="lucene-step">
      <p class="lucene-step-title">3. Carry the late delete into the output</p>
      <div class="lucene-row">
        <span class="lucene-row-label">Input _0</span>
        <span class="lucene-doc lucene-deleted"><s>2 · 103</s><small>Delete applied</small></span>
      </div>
      <div class="lucene-row">
        <span class="lucene-row-label">Output _2</span>
        <span class="lucene-doc lucene-deleted"><s>3 · 103</s><small>Delete carried over</small></span>
      </div>
      <p>The writer sees the delete on _0:2 and marks _2:3 deleted before installing the replacement segment.</p>
    </div>
    <div class="lucene-step">
      <p class="lucene-step-title">4. Refresh the searcher</p>
      <div class="lucene-row">
        <span class="lucene-row-label">Original reader</span>
        <span class="lucene-doc lucene-target">Review 103 found</span>
      </div>
      <div class="lucene-row">
        <span class="lucene-row-label">Refreshed reader</span>
        <span class="lucene-doc lucene-deleted">Review 103 absent</span>
      </div>
      <p>The readers see different snapshots. Both can finish their searches.</p>
    </div>
  </div>
  <div class="lucene-controls" hidden>
    <button type="button" data-merge-action="previous">Previous</button>
    <button type="button" data-merge-action="next">Next</button>
    <button type="button" data-merge-action="play">Play animation</button>
    <span class="lucene-count" role="status" aria-live="polite" aria-atomic="true"></span>
  </div>
</figure>

## How Lucene carries the delete forward

`MergeState` builds a **doc map** for each input. The map tells the codec where to write each surviving document in the output. `SegmentMerger` uses the merge state to write the index structures, including postings, stored fields, and doc values. Review 102 has no output slot because it was deleted before the merge took its view of the inputs. [MergeState](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/MergeState.java) · [SegmentMerger](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/SegmentMerger.java)

While the merge runs, the writer can apply more deletes to the input segments. Before it installs the output, `IndexWriter` compares the inputs' live-doc state at the start of the merge with their current state. It uses the doc maps to carry any newly applied deletes over to the output. The same handoff also carries over applicable doc-values updates. [IndexWriter merge finalization](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java)

That timing leaves four physical slots in the new segment and three live reviews. Review 103 was copied already, so its bytes remain in the output. Search skips that slot. A later merge can reclaim its space.

### Which reader sees the result?

The writer installs the new segment in its current segment list. A near-real-time reader opened from that writer can see the merged index without a durable commit. A reader opened directly from the directory follows the last commit. The original reader from the example keeps its own snapshot even after the writer commits. [DirectoryReader source](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/DirectoryReader.java)

I checked those states with Lucene 10.6.0. The example pauses a forced merge in its merged-segment warmer after the new segment has been written but before publication. It applies the delete, lets the merge finish, and runs a term query for `id=103` against each view.

| View queried | Review 103 | Live docs / physical slots |
| --- | --- | ---: |
| New segment before the writer installs it | Found at doc ID 3 | 4 / 4 |
| Original reader, still open | Found at doc ID 2 | 4 / 5 |
| Fresh writer-backed reader | Absent | 3 / 4 |
| Directory opened before the next commit | Found at doc ID 2 | 4 / 5 |
| Directory opened after the next commit | Absent | 3 / 4 |

Run the [Java example]({{ '/assets/posts/lucene-segment-merges/LateDelete.java' | relative_url }}) with JDK 21 or later and the [Lucene 10.6.0 core JAR](https://repo.maven.apache.org/maven2/org/apache/lucene/lucene-core/10.6.0/lucene-core-10.6.0.jar) in the same directory.

```shell
java -cp lucene-core-10.6.0.jar LateDelete.java
```

## Why disk usage rises before it falls

Lucene writes the output while the input files still exist. After the replacement, an old reader can keep using the inputs. A retained commit can keep old files referenced as well. Space becomes reclaimable once Lucene and the operating system can release those files. [IndexWriter file-lifecycle notes](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java) · [IndexFileDeleter source](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/IndexFileDeleter.java)

<figure class="lucene-figure">
  <figcaption class="lucene-figure-title">Disk space used by the segments in one merge</figcaption>
  <div class="lucene-chart" aria-label="Illustrative disk usage">
    <div class="lucene-chart-row">
      <span>Before the merge</span>
      <div class="lucene-bars" style="--inputs: 5; --output: 0"><span class="lucene-input-bar"></span><span class="lucene-output-bar"></span></div>
      <strong>2.0 GiB</strong>
    </div>
    <div class="lucene-chart-row">
      <span>Inputs and output coexist</span>
      <div class="lucene-bars" style="--inputs: 5; --output: 4"><span class="lucene-input-bar"></span><span class="lucene-output-bar"></span></div>
      <strong>3.6 GiB</strong>
    </div>
    <div class="lucene-chart-row">
      <span>After the inputs are released</span>
      <div class="lucene-bars" style="--inputs: 0; --output: 4"><span class="lucene-input-bar"></span><span class="lucene-output-bar"></span></div>
      <strong>1.6 GiB</strong>
    </div>
  </div>
  <p class="lucene-legend"><span>Input segments</span><span>Output segment</span></p>
  <p class="lucene-figure-note">Illustrative sizes for one merge. Other segments and temporary files can add to the total.</p>
</figure>

## What a merge costs while traffic is live

A completed merge can reduce the number of segments each query visits. During the merge, the writer reads and rewrites index data while queries use the same machine. Lucene 10.6.0 opens separate memory mappings for merge reads and can give those mappings sequential-read advice without changing the advice on the search mappings. The two workloads can still compete for CPU, storage, and page cache. [Lucene 10.6.0 release notes](https://github.com/apache/lucene/releases/tag/releases%2Flucene%2F10.6.0)

`ConcurrentMergeScheduler` runs merges on background threads. If the writer produces segments faster than merges can keep up, the scheduler can stall the incoming thread. An indexing slowdown and a search latency spike can therefore have different timelines even when they come from the same merge backlog. [ConcurrentMergeScheduler source](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/index/ConcurrentMergeScheduler.java)

To see the tradeoff in a workload, chart p99 query latency, indexing throughput, segment count, and disk usage on the same time axis. Mark when each merge starts and ends, and when the searcher refreshes. Hold the offered query rate, update rate, and refresh interval steady. The merge end marks when Lucene finishes the rewrite. The refresh marks when queries can start using the replacement. The old files may remain longer than either. [Lucene near-real-time benchmark](https://benchmarks.mikemccandless.com/nrt.html)
