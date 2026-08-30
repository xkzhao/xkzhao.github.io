---
layout: post
title: "Deep Dive into Apache Lucene's Search Path: From a Query to the Top 10 Hits"
date: 2026-08-30 09:00 -0400
categories: [Search, Database]
tags: [lucene, search-engine, java, ranking, bm25]
---

*Part 2 of the Lucene internals series. [Part 1 — Deep Dive into Apache Lucene's Core: From In-Memory Buffers to Commits](https://kurtzhao.com/posts/deep-dive-into-apache-lucene-core/) followed documents **into** an index. This one follows a query **out** of it.*

Part 1 ended at a durable, committed index: a directory full of immutable segments and a manifest listing them. Which raises the obvious question. You have 2 million documents on disk, someone types `quick dog`, and 5 milliseconds later ten results come back, ranked. Where did the other 1,999,990 documents go?

The short answer is that Lucene never looked at them. The long answer is this article, which is really one idea applied at six different scales: **do not read bytes you don't need, and know in advance which bytes those are.**

**Contents**

1. [A 60-second recap of Part 1](#1-a-60-second-recap-of-part-1)
2. [What "searching" actually means](#2-what-searching-actually-means)
3. [Setup: Opening a reader](#3-setup-opening-a-reader-why-a-500-gb-index-opens-instantly)
4. [Steps 1-2: From query text to a term](#4-steps-1-2-from-query-text-to-a-terms-location)
5. [Step 3: Reading a posting list](#5-step-3-reading-a-posting-list-256-documents-at-a-time)
6. [Steps 4-6: Ranking](#6-steps-4-6-ranking-or-how-lucene-skips-most-of-what-matches)
7. [Steps 7-8: Merging, fetching, adding it up](#7-steps-7-8-merging-fetching-and-adding-it-all-up)
8. [A search performance checklist](#8-a-search-performance-checklist)
9. [Why the design looks like this](#9-why-the-design-looks-like-this)
10. [Caveats and version notes](#10-caveats-and-version-notes)
11. [Appendix: references and code artifacts](#11-appendix-references-and-code-artifacts)

---

## 1. A 60-second recap of Part 1

### The data model: documents and fields

Before the vocabulary, the shape of the data — everything else is built on it.

A **document** is the unit you search for and get back. It is not a file; it's whatever you decided to treat as one searchable thing — a product review, a web page, an email, a row from a database table. In our example, one document is one product review.

A **field** is a named part of a document, like a column in a table or a key in a JSON object. You choose the fields; Lucene attaches no meaning to their names. Our review has three:

```
   one document = one product review
   ├── title    "Great for puppies"                       ← short text, kept for display
   ├── body     "the quick brown fox jumps ..."           ← the searchable text
   └── rating   4                                         ← a number, for sorting or filtering
```

`rating` is here to show a third way of configuring a field; this article's query never sorts on it. Everything below ranks by **relevance**, and §6 explains what that means. Sorting on a field instead is a different code path, sketched at the end of §6.

Each field is configured separately for what you want to do with it, and those choices decide which files in §3 get written:

| what you want | how the field is configured | which file it lands in |
|---|---|---|
| search it | *indexed* — the text is analyzed into terms and inverted | `.tim` / `.doc` |
| display it in results | *stored* — the original value is kept verbatim | `.fdt` |
| sort or filter on it | *doc values* — the value is kept in a column | `.dvd` |

A field can be several of these at once, or none. Our `body` is indexed but not stored (you search it, you don't display the whole thing); `title` is stored; `rating` has doc values.

**Search is always per field.** The term `quick` in `body` and the term `quick` in `title` are two separate entries in the index, with separate posting lists — which is why every query in this article is written `body:quick`, field first.

**"How long a field is" means how many terms it contains**, after analysis — not characters, not bytes. If `body` holds "the quick brown fox", its length is 4. This number is what §6's scoring uses to stop long documents from winning by sheer volume, and it's recorded once per document per field: for segment `_7` that's 524,288 numbers for `body` and another 524,288 for `title`, each squeezed into a single byte:

```
   docID   body length   title length
     0         180            3
     1          42            5
     2         310            4
     ...
```

"One byte per document per field" means one cell of that table costs one byte — so ~512 KB for `_7`'s entire `body` column, small enough to sit alongside the postings while scoring.

### The vocabulary

Five more ideas from Part 1 carry the whole of Part 2. If they are already familiar, skim the table and move on.

| Term | What it means |
|---|---|
| **Inverted index** | Instead of "document → words", Lucene stores "word → documents". Each word is a **term**; the list of documents containing it is its **posting list**. |
| **docID** | A dense integer (0, 1, 2, …) that Lucene assigns to each document *within one segment*. Not your primary key — it changes when segments merge. Every structure in the index is sorted by it. |
| **Segment** | A complete, self-contained mini-index: its own terms, posting lists, stored field values. An index is just a set of segments. |
| **Immutability** | Once written, a segment's files are **never modified**. New documents go into new segments; deletions are recorded in a sidecar bitset (`.liv`) marking docIDs as dead. |
| **`segments_N`** | The manifest: a tiny file listing which segments are live right now. Each commit writes a new one. |
| **Analysis** | Turning text into terms: splitting into tokens, lowercasing, sometimes reducing words to a root. Runs over documents at index time (Part 1) and over the query at search time — and the two must match, or nothing is found. |
| **Codec** | The pluggable set of file formats a segment is written in (`Lucene104Codec` here). It decides how terms, postings, norms and stored fields are laid out on disk — so the byte-level details in this article are codec-specific, while the architecture isn't. |
| **Block** | The next **256 entries of one posting list** — the unit postings are packed and decoded in. Note *entries*, not docIDs: a block holds the next 256 documents **that contain the term**, so it spans as many docIDs as it needs to find them. You cannot read one entry without unpacking its whole block. (The term dictionary also uses "block" for a group of terms sharing a prefix; different thing, same word.) |
| **Region** | 32 blocks — **8,192 posting-list entries**. A coarser navigation level, so a long jump doesn't have to step through hundreds of block signposts. |
| **Bound** (or *ceiling*) | An upper limit on a score, known *without* computing any actual scores — "nothing in this block can score above 5.9". Cheap to read, provably correct, and the basis of all the skipping in §6. |
| **Norms** | One byte per document per field, recording roughly how many terms that field contains. Scoring uses it so a 500-word review doesn't outrank a 50-word one just for mentioning a word more often (§6 works this through). Stored in the `.nvd` file. |
| **`docFreq`** | Document frequency: how many documents contain a given term. Stored next to the term, so it's known before any posting data is read. Distinct from **`freq`**, which is how many times a term occurs *within one document*. |

Immutability is the load-bearing idea. Because a segment's bytes can never change, a reader can map them into memory and use them without any locking, forever — and that single property explains most of the design decisions in the rest of this article.

**The example we'll follow.** One index, one query, all the way through:

- **The index:** 2,000,000 product reviews. The field `body` is searchable, the field `title` is stored for display, the field `rating` is available for sorting. After merges, the index has **12 segments**.
- **The segment we watch:** `_7` — one of the twelve, picked arbitrarily. (Segments are named `_0`, `_1`, … `_a`, `_b` in base 36, as Part 1 described; nothing distinguishes `_7` except that we're following it.) It holds **524,288 documents**, of which 20,971 (4%) have been deleted. In `_7`, the term `quick` appears in **12,000** documents, `dog` in **3,000**, and `the` in **500,000**.

  **A query is executed against every segment, independently.** All twelve run the same steps described below; we follow one of them because the other eleven are the same story with different numbers. §2 shows how their results come back together.
- **The query:** `quick OR dog`, top 10 **by relevance** — not by `rating`. The two-argument `search(q, 10)` ranks by BM25 score, so the sort key is built from term frequency, document frequency and field length, and nothing else.

```java
IndexReader   reader   = DirectoryReader.open(dir);
IndexSearcher searcher = new IndexSearcher(reader);

Query q = new BooleanQuery.Builder()
    .add(new TermQuery(new Term("body", "quick")), SHOULD)
    .add(new TermQuery(new Term("body", "dog")),   SHOULD)
    .build();

TopDocs top = searcher.search(q, 10);
```

About **14,600** documents in `_7` match that query. By the end we'll count how few of them Lucene actually scores.

---

## 2. What "searching" actually means

A search engine answers three separate questions. Keep them apart, because Lucene optimizes each one differently:

1. **Which documents match?** — a set operation over posting lists.
2. **How good is each match?** — a scoring function (BM25 by default).
3. **Which ten do I show?** — a top-K selection.

The naive implementation does them in that order: find all 14,600 matches, score all 14,600, sort, take 10. Lucene interleaves them instead, and uses the answer to question 3 to avoid most of the work in questions 1 and 2. That inversion is the single most important thing to understand about the search path.

Here is the whole thing on one page, as a map for the rest of the article. Each numbered box is a section you can jump to. Before the details, look at two features of the shape: the **fan-out**, because every segment answers the query separately, and the **loop from box 6 back to box 3**, because that feedback is what lets Lucene stop reading data it can prove it doesn't need.

```
                     searcher.search(quick OR dog, 10)
                                      │
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. PREPARE — once for the whole query                                    │
│    analyze the query text into terms; rewrite into primitive clauses     │
│    look up each term's docFreq across ALL 12 segments -> BM25 weights    │
│    quick: 48,000 / 2,000,000 -> idf 3.73     dog: 12,000 -> idf 5.12     │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
      fan out: the same plan runs on every segment, independently
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
       _0             _1             _7             _8            _11
                                      │   ◀ we follow _7; the other 11 are identical
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. LOCATE the term                                            (§4)       │
│    walk the term index (.tip): absent terms are rejected here,           │
│    for free, without touching disk                                       │
│    then read one term-dictionary block (.tim)                            │
│    -> "quick": 12,000 docs; postings start at byte 918,272 of .doc       │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. WALK the posting list                                      (§5)       │  ◀──┐
│    decode .doc in blocks of 256 docIDs                                   │     │
│    signposts give each block's last docID + best possible score          │     │
│    -> skip whole blocks that cannot beat the current 10th-best score     │     │
│    frequencies decoded only when scoring asks; positions only for phrases│     │
├──────────────────────────────────────────────────────────────────────────┤     │
│ 4. DROP deleted documents      live-docs bitset (.liv)        (§3)       │     │
├──────────────────────────────────────────────────────────────────────────┤     │
│ 5. SCORE the survivors    BM25(term freq, field length/norm)  (§6)       │     │
├──────────────────────────────────────────────────────────────────────────┤     │
│ 6. COLLECT into a 10-entry priority queue   (shared by the slice)  (§6)  │     │
│    publishes its 10th-best score back to step 3:                         │     │
│    "nothing below this can win", so step 3 skips more                    │─────┘
└──────────────────────────────────────────────────────────────────────────┘
                                      │
   segments feed one leaderboard per slice (docIDs already offset by docBase)
        └──────────────┴──────────────┴──────────────┴──────────────┘
                                      │
┌──────────────────────────────────────────────────────────────────────────┐
│ 7. MERGE  one top-10 list per slice -> one global top 10                 │
│    docBase was already added at collect time; this is a pure score merge │
├──────────────────────────────────────────────────────────────────────────┤
│ 8. FETCH display fields for those 10 documents only  (.fdt)   (§5)       │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                                  TopDocs
```

**The rest of this article walks these boxes in order**, and each section says which box it is covering:

| box in the map | what it does | where |
|---|---|---|
| — | open the reader the query will run against (once, not per query) | §3 |
| 1. Prepare | analyze the text, rewrite the query, gather term statistics | §4 (analysis, rewrite, statistics), §6 (turning statistics into BM25 weights) |
| 2. Locate | find each term in each segment | §4 |
| 3. Walk | read the posting lists, skipping what can't matter | §5 |
| 4-6. Drop / score / collect | apply deletions, score survivors, keep the best 10 | §6 |
| 7-8. Merge / fetch | combine per-segment results, load the fields to display | §7 |

Mechanically, a query passes through four objects. The layering looks like ceremony until you notice that each boundary exists to delay a decision until more is known:

```
  Query  ──rewrite──▶  Weight  ──per segment──▶  ScorerSupplier  ──▶  Scorer / BulkScorer  ──▶  Collector
    │                    │                          │                      │                      │
 "what to             "index-wide              "how expensive        "walk the matching       "keep the
  match"               statistics"              would I be?"          docIDs, score them"      best 10"
 immutable,          created once per         created per segment,   created per segment,    per segment,
 cacheable           search, holds the        knows cost() before    single-threaded         feeds back a
                     BM25 idf values          doing any real work                            score threshold
```

- **`Query`** is an immutable description — `TermQuery("body", "quick")`. It holds no state and can be cached and reused across searches.
- **`Weight`** binds a query to one `IndexSearcher` and computes index-wide statistics. For our query, this is where Lucene discovers that `quick` occurs in 48,000 of 2,000,000 documents and `dog` in 12,000, and turns those into BM25 weights.
- **`ScorerSupplier`** exists per segment for one reason: it can report **`cost()`**, roughly how many documents this clause will produce, *before* anything commits to an execution strategy. This is where Lucene does its **planning**.
- **`Scorer` / `BulkScorer`** does the actual walking and scoring, one segment at a time, single-threaded. (Concurrency comes from searching different segments — or different slices of one segment — in parallel.)
- **`Collector`** keeps the running top 10, and tells the scorers what a document now has to score to be worth looking at. That last part is what §6 is built on.

**"Planning"** runs through the rest of the article. It means deciding *how* to execute a query, as distinct from *what* the query asks for. The same query can usually be run several ways that return identical results at wildly different cost, and choosing between them is planning.

The clearest example is `quick AND dog`. Two ways to execute it:

```
   (a)  walk quick's 12,000 documents, check each one against dog   -> 12,000 steps
   (b)  walk dog's  3,000 documents, check each one against quick   ->  3,000 steps
```

Same answer, a quarter of the work. Lucene picks (b), and it can only do so because §4's lookup already handed it both list sizes for free. Other decisions of the same kind appear later: whether a clause gets walked or merely consulted (§6), whether to score at all or just match, whether to build a bitset instead of iterating, which of a dozen scorer implementations to instantiate.

Two things distinguish this from a SQL planner. Lucene plans **per segment**, so the same query can execute differently in `_7` than in `_9`; and it can **re-plan while running**, because its cost inputs stay cheap to obtain. What it lacks is an `EXPLAIN` — the plan isn't a printable tree, it's the set of scorer objects that got constructed.

That last arrow in the diagram, running backwards from collector to scorer, is where most of the interesting behaviour comes from. It reappears in §6.

### Every segment is searched separately, then merged

The fan-out at the top of that map deserves its own diagram, because it shapes everything else, including two things that catch people out later: segment-local docIDs, and why segment count is a performance number.

```
                  ┌──────────────┐
                  │    Query     │   parsed and planned once
                  └──────┬───────┘
                         │ Weight: index-wide statistics (IDF) computed once,
                         │         across all 12 segments
         ┌───────────────┼───────────────┬─────────────┐
         ▼               ▼               ▼             ▼
      segment _0      segment _7      segment _8   …  _11        ← independent,
      own docIDs      own docIDs      own docIDs                   no shared state,
      own postings    own postings    own postings                 parallelizable
         └───────────────┴───────────────┴─────────────┘
                         │  segments in the same slice share one 10-entry
                         │  leaderboard, so the bar carries across them
                         │  merge: one top-10 per slice → one global top 10
                         ▼
                      TopDocs
```

Three consequences follow, and all three surprise people later:

- **docIDs are segment-local.** Document 42 in `_7` and document 42 in `_8` are different documents. The final `TopDocs` reports global identifiers, produced by adding each segment's `docBase` offset — the collector does this as it collects, so the per-segment queues already hold index-wide docIDs and the merge is a pure comparison of scores. This is also why docIDs aren't stable across merges and must never be used as primary keys.
- **Segment count is a per-query multiplier.** Twelve segments mean twelve term lookups and twelve posting-list walks. The work per segment is proportional to the data in it, so splitting a corpus into more segments doesn't change the total posting data read, but it does multiply the fixed per-segment costs. That's why merging matters for search latency and not only for disk usage.
- **The leaderboard is per *slice*, not per segment.** This one catches people out. A slice is the unit of concurrency, and with no executor there is exactly one: `IndexSearcher` builds `LeafSlice.entireSegments(leafContexts)`, and `newCollector()` is called once per slice. So a default single-threaded search over our twelve segments has **one** 10-entry heap, **one** `totalHits` counter and **one** bar for the whole query. The bar carries across the segment boundary — `_8` starts out with whatever `_0` through `_7` built up, and only the first segment pays for warming it up. Hand the searcher an executor and you get several slices (at most 5 segments or 250,000 documents each), each with its own heap and its own bar, which is when the merge in §7 has more than one list to merge.
- **This is where concurrency comes from.** Because segments share no state, Lucene searches them in parallel when given an executor, grouping small segments together and, for a large segment like our `_7`, optionally splitting it into ranges searched concurrently. Within one segment (or range), execution is strictly single-threaded.

One statistic deliberately does *not* work per segment: the BM25 IDF values in §6 are computed across the whole index before any segment is touched. If each segment scored with its own local term frequencies, the same document would score differently depending on which segment it happened to land in.

---

## 3. Setup: Opening a reader (why a 500 GB index opens instantly)

```java
IndexReader reader = DirectoryReader.open(dir);
```

Two things about this line surprise most people. It takes milliseconds regardless of index size, and the view it returns is frozen in time.

### It's a snapshot

`DirectoryReader.open` reads the newest `segments_N` manifest and opens the segments it lists. Because those segment files can never be modified, the resulting view stays consistent no matter what the writer does next — new commits, merges, deletions, all invisible until you deliberately reopen. There are no locks and no coordination with the writer. A long-running search and a heavy indexing job simply don't interact.

(The one race Lucene must handle: between listing the directory and opening `segments_9`, a commit plus cleanup can delete it. The open path retries with the newer generation rather than taking a lock — the only retry loop in the entire read path.)

### It's metadata-only

Opening a segment does **not** read the index data. For each segment, and each kind of data in it, Lucene reads a small *metadata* file and memory-maps the *data* files:

```
segments_9   ← the manifest: 12 live segments
   ├── _0 … _6      (opened, metadata only)
   ├── _7           ← the segment we follow
   │     ├── CORE — immutable, safe to share:
   │     │     .tim/.tip  term dictionary + term index   (mapped; nothing read yet)
   │     │     .doc/.pos  posting lists + positions      (mapped; nothing read yet)
   │     │     .nvd       norms — 1 byte/doc: how long the field is
   │     │     .dvd       doc values (our `rating` column)
   │     │     .fdt       stored fields (our `title` values)
   │     └── NOT CORE — changes with every delete:
   │           _7_3.liv   live-docs bitset: 20,971 of 524,288 marked dead
   └── _8 … _11
```

So the cost of opening a reader is roughly *(number of segments) × (number of fields)*. For us that's 12 segments and a handful of fields, a few milliseconds, and **not** a function of index size. The flip side: an index with 10,000 tiny segments is slow to open and slow to search, no matter how small it is. Segment count is a first-class performance number.

> **Under the hood.** `index/StandardDirectoryReader.java` drives the open through `SegmentInfos.FindSegmentsFile` (the retry loop above). Per-segment work lives in `index/SegmentCoreReaders.java`, which asks the codec for a `FieldsProducer`, `NormsProducer`, `StoredFieldsReader`, and so on. Each producer reads its metadata file, then keeps `IndexInput`s open on the data files. For the term dictionary, `_7.tmd` gives the term count per field and the offsets of that field's slice of the term index.

### The bytes come from the OS, not the heap

On 64-bit platforms the default `Directory` is `MMapDirectory`: index files are mapped into the process address space, so reading a posting list is a memory access, not a syscall. Page faults do the I/O and **the operating system's page cache is the index cache**.

Which leads to two pieces of advice that matter more than any tuning flag:

- **Leave RAM free.** An oversized JVM heap starves the page cache and turns memory reads into disk reads. Lucene deployments are usually happiest with a modest heap and lots of free RAM.
- **First query after a reopen is the slow one**, because nothing is resident yet. That's what warming is for.

> **Under the hood.** `store/MMapDirectory.java` and `store/MemorySegmentIndexInput.java` use the Panama FFM API. `IOContext` hints become a `store/ReadAdvice.java` value (`NORMAL`, `RANDOM` or `SEQUENTIAL`), which `store/PosixNativeAccess.java` maps onto a `posix_madvise` call; `MMapDirectory.setPreload(...)` forces pages resident at open time. There is also an explicit `prefetch(offset, length)` primitive, which issues `POSIX_MADV_WILLNEED` for the range. Remember it: "tell the OS what you'll need, then go and get it" turns up three more times below.

### Reopening pays only for what changed

Because segments are immutable, a reopen is cheap in proportion to what actually changed. `DirectoryReader.openIfChanged(reader)` returns `null` if there's been no commit at all; otherwise unchanged segments are shared with the old reader by reference, a segment that only gained deletions gets a fresh live-docs bitset over its still-warm data, and only brand-new segments are really opened. Adding one small segment to our twelve-segment index costs one segment open, not twelve.

The same mechanism, pointed at an `IndexWriter` instead of a directory, gives **near-real-time search**: `DirectoryReader.open(indexWriter)` makes the writer flush its in-memory buffers and hand out readers over segments that were never committed. Visibility moves from *commit* (expensive, `fsync`) to *flush* (cheap) — that's the whole trick behind "index now, search in 50 ms".

In a server, don't manage any of this by hand: `SearcherManager` plus `ControlledRealTimeReopenThread` handle the reference counting and hand fresh readers to new requests without yanking one out from under a query in flight.

---

## 4. Steps 1-2: From query text to a term's location

### Step 1a: turning query text into terms

Before anything can be looked up, the text the user typed has to be converted into the exact byte sequences stored in the index. That conversion is called **analysis**, and it is the same process Part 1 ran over documents at index time: split the text into tokens, lowercase them, possibly reduce them to a root form.

```
   user types:   "Quick DOGS!"
   analysis:      lowercase, split on non-letters, strip plurals
   terms:         [ quick ] [ dog ]        ← these are what the index contains
```

The rule that follows is the single most common source of "why does my search return nothing?": **the query must be analyzed the same way the field was.** If documents were indexed with a lowercasing analyzer and the query isn't, looking up `Quick` finds nothing at all, because the index only ever contained `quick`. (Fields indexed as exact keywords, such as IDs, enum values and tags, are deliberately *not* analyzed on either side.)

### Step 1b: rewriting the query into primitive clauses

Not every query can be executed directly. A wildcard like `qui*`, a fuzzy match, or a numeric range doesn't name a term — it *describes* a set of them. **Rewriting** turns those into something executable, repeatedly, until nothing changes:

```
   qui*   ──rewrite──▶   quick OR quicken OR quicker OR quickly
                         (found by walking the term index below;
                          if too many terms match, it becomes a
                          single clause that builds a bitset instead)
```

Rewriting also simplifies: a one-clause boolean query unwraps to its clause, a constant-score wrapper drops scoring machinery it doesn't need. Two things about it matter. It happens **once per query, before any segment is searched**. And it is where a badly-shaped wildcard reveals its cost: `a*` on a large field expands to an enormous number of terms right here, which is how leading-wildcard and very short prefix queries earned their reputation.

After analysis and rewriting we hold something concrete: two `TermQuery` clauses, `body:quick` and `body:dog`. Now they have to be located.

### Step 2: finding the term in each segment

Now segment `_7` has to answer: where are the documents containing `body:quick`?

A term dictionary with 900,000 distinct terms is far too large to keep in RAM as a hash map, and too large to binary-search on disk without many random reads. Lucene splits it in two:

```
  "quick"
     │
     │  (a) the term index (.tip): a prefix trie — a tree whose edges are
     │           single bytes, so a path through it spells a prefix — mapped
     │           into memory. Walk it one byte at a time.
     ▼
   q ─ u ─ i ─ c ─ k  ──▶  "the block you want begins at byte 4,182,208 of .tim"
     │
     └─ "quizzical"? There is no 'z' child under "qui".
        Answer: the term does not exist. Zero disk reads.
     │
     │  (b) the term dictionary (.tim): one block, read once, scanned.
     ▼
   [ quick | quicken | quicker | quickly ]
        └─▶ quick: appears in 12,000 docs; its posting list starts at byte 918,272 of .doc
```

Think of it as a library: the term index is the catalog that tells you which shelf to walk to, and the term dictionary is the shelf. Terms sharing a prefix live in the same block, so the catalog only has to be precise down to block granularity — which is what keeps it small enough to stay resident.

The property to remember is the one in the middle of the diagram: **Lucene can usually prove a term is absent without touching the dictionary at all.** For a user's typo, or a term that only exists in 2 of 12 segments, the other 10 segments reject it during the trie walk, in mapped memory, for free. This matters enormously for queries with many terms, and for `id`-field lookups across many segments.

### What the lookup hands back: docFreq

Reading the dictionary entry for `quick` costs one block read, and it yields more than a file pointer. Sitting right next to the term is its **`docFreq`** — short for *document frequency*: **the number of documents that contain this term at least once.**

Pin that down, because Lucene has three similar-sounding counts that do completely different jobs:

| name | scope | what it counts | our example |
|---|---|---|---|
| **`docFreq`** | one term | how many **documents** contain the term | `quick`: 12,000 documents in `_7` |
| **`freq`** (term frequency) | one term **in one document** | how many **times** the term occurs in that document | document 41,502 contains `quick` 3 times |
| **`totalTermFreq`** | one term | total occurrences across all documents | `quick`: maybe 15,500 occurrences in `_7` |

`docFreq` is simply the length of the posting list — how many entries the term has. What makes it valuable is *when* you learn it: the term dictionary stores it explicitly, so it's known **immediately after the lookup, before a single byte of the posting list is read.** A cheap number, available early, that describes the size of expensive work you haven't started yet. Two uses follow directly:

- **Planning.** `docFreq` is the iterator's `cost()`. When §6 decides to lead an AND query with `dog` rather than `quick`, it's comparing 3,000 against 12,000 — numbers it got for free.
- **Scoring.** Summed across all 12 segments, `docFreq` gives the corpus-wide rarity of the term: `quick` in 48,000 of 2,000,000 documents, `dog` in 12,000. That's the input to BM25's IDF, computed in §6.

(A word of caution if you ever print these: `docFreq` counts deleted documents too, since deletions are recorded outside the term dictionary. It's an upper bound, which is exactly what a planner wants.)

> **Under the hood.** `codecs/lucene103/blocktree/`. The term index (`.tip`) is a prefix trie (`TrieReader`) whose nodes carry a label and, at block boundaries, a file pointer into `.tim`; older Lucene versions used an FST here. The dictionary (`.tim`) stores terms in blocks sharing a prefix, each entry carrying `docFreq`, `totalTermFreq`, and file pointers into `.doc`/`.pos`. `SegmentTermsEnum` keeps the trie walk on a frame stack and reuses the common prefix between consecutive seeks — which is why enumerating `quic*` costs far less than four independent lookups.
>
> One subtlety with outsized payoff: `prepareSeekExact()` performs the trie walk but returns a *supplier* for the part that may need disk. `index/TermStates.java` collects suppliers for all terms across all segments first, then resolves them — so 24 term lookups (2 terms × 12 segments) issue their I/O concurrently instead of one after another. Same idea as `prefetch`, one layer up.

---

## 5. Step 3: Reading a posting list, 256 documents at a time

§4 gave us a pointer into the `.doc` file and a promise: 12,000 documents contain `quick`. Time to read them.

A posting list is, conceptually, nothing but a sorted list of docIDs:

```
   "quick" appears in documents:   7, 51, 98, 132, 180, 205, ... (12,000 of them)
```

Stored as plain 4-byte integers that's 48 KB per term, and the only way to find a particular docID in it is to read from the start. Lucene fixes both problems, size and navigation, with four ideas. Each is simple on its own; the combination is what does the work.

### Idea 1: store the gaps, not the numbers

The docIDs are ascending, so the differences between them are much smaller than the numbers themselves:

```
   docIDs stored:    7    51    98   132   180   205
   gaps written:     7    44    47    34    48    25      ← all fit in 6 bits
```

Reconstructing docIDs is a running sum. In our segment, `quick` appears in 12,000 of 524,288 documents, so gaps average about 44 and nearly all fit in 7 bits — about a quarter of the space of raw integers.

### Idea 2: pack the gaps in groups of 256

Bit-packing needs a width, and picking one width per gap costs more than it saves. So Lucene groups gaps into **blocks of 256**, finds the widest gap in each block, and packs the whole block at that width. One byte at the front of the block records the width.

This makes decoding a bulk operation: unpack 256 gaps in one tight, vectorized loop, run a prefix sum, and you have 256 docIDs in an array. The cost is that a block is all-or-nothing — you can't read document #300 without decoding the block that contains it. That trade is heavily in Lucene's favor, because decoding a block is on the order of a few hundred CPU instructions and no I/O beyond the bytes themselves.

`quick`'s 12,000 documents therefore live in 46 full blocks plus a 224-document tail. The tail is written as **variable-length integers**, small numbers taking one byte and larger ones more, since a partial block isn't worth bit-packing.

### Idea 3: put signposts between the blocks

Now the navigation problem. Queries rarely want "the next document" — they want *"the first document at or after 300,000"*, because some other clause has already jumped ahead. Without help, answering that means decoding every block along the way.

So between blocks, Lucene writes a small **signpost**:

```
 .doc file, posting list for "quick"

  ┌─ signpost ─┐┌─── block 0 ───┐┌─ signpost ─┐┌─── block 1 ───┐        ┌── tail ──┐
  │ last docID ││ 256 docID gaps││ last docID ││ 256 docID gaps│  ...   │ 224 docs │
  │  = 4,712   ││               ││  = 9,530   ││               │        │ as vints │
  │ max score  ││ 256 freqs     ││ max score  ││ 256 freqs     │        └──────────┘
  │  = 5.9     ││               ││  = 4.1     ││               │
  └────────────┘└───────────────┘└────────────┘└───────────────┘
        ▲
        └── read this (a few bytes) to decide whether to decode the block at all
```

The first field, "last docID in the next block", is a classic skip list. To answer *advance to 300,000*, Lucene reads signposts, never blocks, until it finds the first one whose last docID reaches 300,000; then it decodes that one block and searches inside it. A second, coarser level of signposts lands every 32 blocks, or 8,192 posting-list entries, which this article calls a **region**. Long lists therefore don't require walking every fine-grained signpost either. A jump forward consults region signposts first, then block signposts, then decodes one block, so jump distance barely affects cost.

(Terms with fewer than 8,192 documents never reach the coarse level at all. `dog`, with 3,000, has a posting list smaller than a single region, though it still gets a fine-grained signpost per 256-entry block. Only a term small enough to fit in one partial block, under 256 documents, carries no skip data whatsoever.)

**A block is 256 entries, not 256 docIDs.** It holds the next 256 documents *that contain the term*, so how far it reaches across the segment depends entirely on how common that term is:

| term | docs in `_7` | docIDs spanned by one block |
|---|---|---|
| `the` | 500,000 | ~268 |
| `quick` | 12,000 | ~11,200 |
| `dog` | 3,000 | ~44,700 |

A region is 32 blocks, so the same caveat applies with a bigger number: 8,192 entries, not 8,192 docIDs. `dog`'s entire posting list is smaller than one region, which is why it never gets a coarse signpost.

This is also why the scorers in §6 work in windows whose edges keep moving. They ask a posting list where its current block ends, get a docID back, and that answer is different for every term in the query.

On disk the arrangement is what the diagram above suggests — signposts and data interleaved in one stream, nothing off to the side. The format javadoc writes the grammar as:

```
   TermFreqs      →  <PackedBlock32>^(docFreq/8192) , VIntBlock?
   PackedBlock32  →  Level1SkipData , <PackedBlock>^32
   PackedBlock    →  Level0SkipData , PackedDocDeltaBlock , PackedFreqBlock?
```

A region signpost, then 32 repetitions of (block signpost, packed docIDs, packed frequencies). Because every signpost immediately precedes the bytes it describes, reading one never costs a seek, and `Level0SkipData` carries a `PackedBlockLength` so a reader can hop over the body without decoding it.

Blocks aren't aligned to anything the hardware cares about, though, and it's easy to undercount them by looking only at docIDs. `quick`'s 256 gaps at 7 bits are 224 bytes, but the frequencies live in the same span, taking up room even on the reads that never decode them, and the signpost's impacts sit on top of that. Call it 325 bytes, so roughly a dozen blocks to a 4 KB page.

Which is the real difference between the two levels. Skipping a block saves the unpacking, the prefix sum and the scoring, but the bytes you skipped were almost certainly on a page you faulted in anyway — it buys CPU, not I/O. A region is ~10 KB and spans several pages, so skipping one is a genuine I/O saving. On a cold index that's where the milliseconds are.

### Idea 4: the signpost also says how good the block could be

The second field is the one that makes §6 possible. Along with "last docID", each signpost carries the **best score any document in that block could possibly achieve** — Lucene calls these *impacts*.

This is only possible because BM25 is well-behaved: a document's score for a term depends on exactly two per-document numbers — how often the term appears in it, and how long the field is. Store the highest frequency and the shortest length seen in a block, run the scoring formula on that combination, and you have an upper bound on the entire block, valid for every document in it. (Lucene stores something slightly sharper than that pair; §6 gets to it.)

So before touching a block, the engine can ask:

> *"The best any of these 256 documents can score is 5.9. My current 10th-best result scores 7.2. Is it worth decoding?"*

No. Skip 256 documents without reading them. That question costs a few bytes to answer, and it's the entire foundation of the next section.

### And it reads nothing you didn't ask for

Look at the layout again: each block stores docIDs *and* frequencies. When Lucene decodes a block of docIDs, it notes where the frequencies are and steps over them. They're decoded only if someone actually calls `freq()` — which happens only when a document is genuinely being scored.

Positions (which word slot the term occupied, needed only by phrase queries) aren't in this file at all; they live in `.pos` and are read only when a phrase check asks for them.

The effect is large and easy to miss. For `+quick +dog`, where `quick` is only ever checked at candidate docIDs, most of `quick`'s frequency data is never decoded — close to a quarter of the posting file, never touched. For the phrase `"quick brown"`, positions are read only for documents containing both words, not for the 12,000 containing `quick`.

### One more encoding, for very common terms

Gap encoding assumes gaps are worth encoding. For `the`, which appears in 500,000 of `_7`'s 524,288 documents, nearly every gap is 1 — and a list that dense is better stored as a **bitset**: one bit per document, set if the document contains the term. Lucene picks per block, using that same leading byte:

| term | docs in `_7` | typical gap | what a block looks like |
|---|---|---|---|
| `dog` | 3,000 | ~175 | ~9 bits per gap, bit-packed |
| `quick` | 12,000 | ~44 | ~7 bits per gap, bit-packed |
| `the` | 500,000 | ~1 | a bitset — and when 256 docIDs happen to be consecutive, literally one byte for the whole block |

Which is why stop-words aren't the disaster they were in older search engines: in modern Lucene, `the` costs *less per document* than a mid-frequency term. You generally shouldn't strip them.

> **Under the hood.** `codecs/lucene104/Lucene104PostingsReader.java`. Blocks are `BLOCK_SIZE = 256`; the coarse signpost level lands every `32 × 256 = 8,192` postings (`LEVEL1_NUM_DOCS`), counted per term (`docCount` is reset in `startTerm`, so region boundaries are positions in the posting list, not docIDs). `refillFullBlock()` reads one byte to choose the encoding: positive = bit-packed gaps plus a vectorized prefix sum, negative = a bitset of that many longs, zero = 256 consecutive docIDs. Frequencies are skipped by recording `freqFP` and calling `PForUtil.skip()`, then decoded on the first `freq()` call. The signposts are `advance()`'s skip data; the score bounds are reached through `ImpactsEnum#advanceShallow` / `getImpacts`, which position the skip readers without decoding any postings.

### One last thing this file doesn't contain

Nothing read so far is text. Posting lists, norms and live docs are all the engine needs to decide *which* documents win; the actual `title` string of a review lives in a separate, compressed file and is not touched until §7, once the winners are known. Keeping display data out of the matching path is deliberate — it is the difference between decompressing 10 documents and decompressing 14,600.

---

## 6. Steps 4-6: Ranking, or how Lucene skips most of what matches

Our query matches ~14,600 documents in `_7`. We want the best 10. The naive plan is to score all 14,600 and sort. Here is how Lucene ends up scoring a small fraction of them — and, importantly, still returns *exactly* the same ten documents.

### First, what a score is

Lucene's default relevance model is BM25. A document scores higher for a term when:

- the term appears **more often** in the document (term frequency),
- the term is **rare in the corpus** (inverse document frequency, IDF),
- the document's field is **short** rather than long (norms).

The corpus-wide part, IDF, is computed once before any segment is touched, from two numbers that §4 collected for free while looking the terms up: how many documents contain the term (`docFreq`), and how many documents there are in total. Lucene's formula is

```
                     N − docFreq + 0.5
    idf  =  ln( 1 + ─────────────────── )         N = documents in the index
                       docFreq + 0.5
```

Plugging in our corpus of N = 2,000,000 reviews:

```
    quick   docFreq = 48,000   ->  ln(1 + 1,952,000.5 / 48,000.5)  =  ln(41.67)  =  3.73
    dog     docFreq = 12,000   ->  ln(1 + 1,988,000.5 / 12,000.5)  =  ln(166.66) =  5.12
```

The shape of that curve explains a lot of ranking behaviour:

| a term in… | docFreq | IDF |
|---|---|---|
| every document | 2,000,000 | 0.00 |
| half the corpus | 1,000,000 | 0.69 |
| 10% of the corpus | 200,000 | 2.30 |
| `quick` — 2.4% | 48,000 | 3.73 |
| `dog` — 0.6% | 12,000 | 5.12 |
| 0.05% | 1,000 | 7.60 |
| a single document | 1 | 14.10 |

A term that appears everywhere contributes **nothing** — which is the principled version of a stop-word list, and another reason modern Lucene doesn't need one. Rarity is rewarded steeply but with diminishing returns, since the logarithm keeps a one-off typo from swamping the score.

So `dog`, the rarer word, is worth about 37% more than `quick` on its own, and a document containing only `quick` starts out behind. Lucene is about to exploit that ruthlessly.

Two practical notes. `N` here is the number of documents that actually have the field being searched, not the raw document count, and both numbers are summed across all 12 segments before the search fans out (the reason §7's merge is legitimate: every segment scored with the same IDF). And the two guards in the formula matter: the `+ 0.5` terms keep it defined for a term in every document or none, and the `1 +` keeps IDF at or above zero — textbook BM25 without it goes *negative* for any term in more than half the corpus, which would mean matching a word actively hurt a document.

For a two-clause OR query, a document's score is simply the sum of whichever clauses it matches.

### Where the ceilings come from

The next two sections lean on statements like *"a document matching `quick` alone can score at most 5.9."* That number isn't a guess, and it isn't computed by looking at documents at search time — it's read out of the index.

A document's BM25 score for one term depends on exactly **two per-document numbers**: how many times the term occurs (`freq`), and how long the field is (the norm). And the formula is monotonic in both — more occurrences can only raise the score, a longer field can only lower it:

```
                                freq × (k1 + 1)
    score  =  idf  ×  ─────────────────────────────────────
                       freq + k1 × (1 − b + b × len/avgLen)          k1 = 1.2, b = 0.75
```

Two of those symbols need unpacking.

**`freq`** is easy: how many times the term occurs in this document.

**`len` and `avgLen` are about document length**, and they exist to fix an unfairness. Consider two reviews that both mention `dog` three times:

| | length of the `body` field | mentions of `dog` |
|---|---|---|
| review A | 50 words | 3 |
| review B | 500 words | 3 |

Review A is *about* dogs. Review B mentions dogs in passing — in 500 words, three mentions is close to accidental. But `freq` is 3 for both, so on frequency alone they'd score identically. Longer documents win by default simply for being longer, which is not what anyone wants.

So BM25 compares each document's length against the corpus average. `len` is this document's field length (how many terms it contains after analysis), `avgLen` is the average across the corpus — say 200 words for our reviews. The ratio says whether this document is short or long *relative to typical*:

```
   review A:   len/avgLen = 50/200  = 0.25    ← a quarter of typical length: short
   typical:    len/avgLen = 200/200 = 1.0
   review B:   len/avgLen = 500/200 = 2.5     ← two and a half times typical: long
```

That ratio sits in the denominator, so a longer-than-average document divides its score down. Running all three through the formula with `idf = 5.12` and `freq = 3`:

| | len/avgLen | score for `dog` |
|---|---|---|
| review A (50 words) | 0.25 | **9.6** |
| an average review (200 words) | 1.0 | **8.0** |
| review B (500 words) | 2.5 | **6.1** |

Same three mentions, a 57% spread — decided entirely by length.

The two constants control how hard each effect bites. **`k1`** governs how quickly repeated occurrences stop helping: at `k1 = 1.2`, going from one mention to two matters a lot, from nine to ten almost nothing. **`b`** governs how much length is penalized: `b = 0` ignores length entirely (the table above would collapse to one number), `b = 1` normalizes fully; `b = 0.75` is the usual middle.

`len` isn't stored as an exact number. At index time Lucene compresses each field's length into a single byte per document per field, called a **norm** — that's the `.nvd` file from §3. It's lossy, which is fine: the difference between a 300-word and a 305-word review doesn't change anyone's ranking, and one byte per document per field keeps norms small enough to read alongside the postings without a second thought.

### From per-document scores to a ceiling

Now the step that matters. Notice that the formula only ever moves in one direction with each input: **more occurrences can only raise a score, a longer field can only lower it.** Nothing else varies — `idf`, `k1`, `b` and `avgLen` are the same for every document.

So if you want to know the highest score *any* document in a group could get, you don't need to score them all. You need the most favourable combination of `freq` and length that occurs in the group.

Here's that idea on real documents. These are four of the 12,000 reviews containing `quick` in segment `_7` (`idf = 3.73`, average review = 200 words):

| docID | times `quick` occurs | body length | len/avgLen | score |
|---|---|---|---|---|
| 41,001 | 1 | 300 words | 1.5 | 3.10 |
| 41,502 | **3** | **200 words** | **1.0** | **5.86** |
| 41,780 | 2 | 900 words | 4.5 | 2.58 |
| 42,110 | 1 | 120 words | 0.6 | 4.46 |

Document 41,502 wins this group: three occurrences is the highest frequency here, and its length is unremarkable. Its score, 5.86, is therefore a **ceiling** for these four documents — nothing else among them can beat it.

Extend that from four documents to the whole 8,192-entry region these four sit in, and you get the sentence §6 leans on: **across this region of `quick`'s posting list, the most favourable combination that occurs is 3 occurrences in an average-length body — worth 5.86, call it 5.9.** No document containing only `quick` can score higher within this region. That is what "the strongest thing the index recorded" means: not a theoretical maximum, but the best combination that genuinely appears in the data.

**When does that happen?** Two different moments are mixed together above, and separating them answers most questions about how this can possibly be cheap.

The *combination*, 3 occurrences in a 200-word body, is found **while the index is being written**. As the posting list for `quick` is laid down, every document's frequency and field length pass under the writer's nose anyway, so noting the best ones costs nothing extra. That much is baked into the segment and never changes.

The *score*, 5.86, is **not** stored and could not be. It depends on `quick`'s idf, which is a property of the whole index and shifts as documents are added, and on which similarity you configured, with which `k1` and `b`. None of that is known when the segment is written. So the 5.86 in the table above is query-time arithmetic; I showed it as a score only to make the idea concrete.

The ceiling is therefore half index-time and half query-time, and the next section is about where that seam falls.

The bound is also regional rather than segment-wide. A different region of the same list may hold a document with 9 occurrences in a short field, and it will have its own, higher ceiling. Nothing breaks — the engine re-derives the ceiling for every window it enters.

The same exercise for `dog`, where `idf = 5.12` and the best combination that occurs is 5 occurrences in a body about three-quarters of average length, gives:

```
                     5 × (1.2 + 1)                   11
    5.12  ×  ───────────────────────────  =  5.12 × ─────  =  9.43     ->  9.4
              5 + 1.2 × (0.25 + 0.75×0.75)           5.975
```

### What is actually stored (and when)

That derivation is the definition of the ceiling — score every document in a group, take the maximum. It isn't what Lucene literally does.

**The index cannot store a score.** A score depends on things unknown at index time: the IDF (which is summed across all 12 segments, and changes as the index grows) and the scoring model itself (BM25 today, something else if you configure it, with whatever `k1` and `b`). Storing 5.9 would freeze all of that.

So the index stores the **inputs**, not the answer. While writing the posting list, when it is looking at every document anyway, Lucene keeps for each block the small set of `(freq, length)` pairs that could possibly produce the best score, and discards the rest. A pair goes when another beats it on both counts: 2 occurrences in a 500-word review can never outscore 3 occurrences in a 200-word one, so there is no reason to keep it. What survives is a handful of combinations that genuinely compete, like "3 in a short field" and "8 in a long field", none of which dominates the others.

Then, at query time, computing the ceiling is trivial: run the current scoring formula over those few stored pairs and take the maximum. That's a handful of arithmetic operations on data already in hand, not 12,000 documents examined:

```
   index time   ->  per block, keep the winning (freq, length) pairs      e.g. (3, 1.0×avg), (8, 3.0×avg)
   query time   ->  score each pair with this query's idf and similarity  ->  5.86, 4.90
                    take the max                                          ->  ceiling = 5.9
```

The result is exactly the maximum of the real per-document scores, because the discarded pairs were provably worse — the pruning is safe for the same monotonicity reason the whole scheme is.

These pairs are recorded at exactly **two** granularities, matching §5's two levels of signposts — a bound over fewer documents is a tighter bound:

| granularity | what's stored | the ceiling it yields is used by |
|---|---|---|
| one block (256 docs) | the winning `(freq, length)` pairs among those 256 | Skip 1 — "decode this block or not?" |
| one region (8,192 entries) | the same, for the region | coarse jumps, and the window ceilings of Skip 2 |

There is **no whole-posting-list ceiling in the index.** If you ask for a bound over a range wider than one region, Lucene has nothing recorded and falls back to a purely theoretical bound — the score of a document with infinite frequency and the shortest possible field, which for BM25 is just `idf × (k1 + 1)`: 8.2 for `quick`, 11.3 for `dog`. That number is nearly useless for pruning, which is precisely why the engine works in **windows** bounded by block and region boundaries (below) rather than trying to reason about whole lists.

So when this article says "`quick`'s ceiling is 5.9", read it as *the ceiling over the region the engine is currently looking at* — the number it actually gets from `getMaxScore(windowEnd)`:

| a document that matches… | the best it could possibly score | where that comes from |
|---|---|---|
| `quick` only | **5.9** | idf 3.73, best combination in the data: 3 occurrences, average-length field |
| `dog` only | **9.4** | idf 5.12, best combination in the data: 5 occurrences, short field |
| both | **15.3** | the two ceilings added |

These are facts about your data, not theory. If no review ever mentions `quick` more than three times, the ceiling says so; a different corpus gives different numbers. They are also intensely local — each of the 12 segments records its own impacts, and within a segment so does every block and every region, so a "ceiling" is always a claim about one stretch of one posting list, never about the term.

And they tighten as you zoom in. `quick`'s ceiling over a whole 8,192-entry region may be 5.9 while most blocks inside it sit well below it — the block in §5's diagram bottoms out at 4.1. Skip 2 uses the looser region number to decide how to organize the clauses for the window ahead; Skip 1 uses the tight per-block numbers to decide which parts of that window to decode.

> **Under the hood.** The stored pairs are called *impacts*. They live in `index/FreqAndNormBuffer.java` as parallel `int[] freqs` / `long[] norms` arrays — literally the two inputs, no score. `codecs/CompetitiveImpactAccumulator.java` is what prunes them at index time, keeping only the Pareto frontier: a pair is dropped when another has both a higher freq and a more favourable norm. `search/MaxScoreCache.java#computeMaxScore` runs the similarity over the surviving pairs and takes the maximum; `Scorer#getMaxScore(upTo)` is the public way to ask for a ceiling, and `IndexSearcher#explain` shows how a real score decomposed.
>
> `Impacts#numLevels()` returns **1 or 2**, block and region, and never more. `MaxScoreCache#getMaxScore` falls back to `globalMaxScore = scorer.score(Float.MAX_VALUE, 1L)` when `upTo` outruns both levels, which is the `idf × (k1 + 1)` theoretical bound mentioned above.

Keep that ceiling table in view. Everything below is one consequence of it.

### The bar

The collector keeps a 10-entry leaderboard, and the **bar** is the score of its weakest entry: anything below that cannot reach the final answer.

Nothing computes the bar. The leaderboard is a min-heap whose root *is* the weakest of the ten, so reading the bar means reading the root. The whole structure is ten `long`s: `DocScoreEncoder` packs the score into the high 32 bits and `Integer.MAX_VALUE - docID` into the low 32, so a single `long` comparison orders by score and breaks ties toward lower docIDs, with no objects and no comparator.

That makes the per-document cost tiny. The collector compares each new score against the root; below it, the document is dropped after one float comparison, which on a query matching 14,600 documents is what happens to almost all of them. Above it, the root is overwritten and sifted down, three comparisons in a ten-entry ternary heap.

Two details matter later. The heap starts **pre-filled with ten sentinels scoring −∞**, so it is never "not yet full" — what gates the bar is the hit threshold, not the leaderboard. And the value published is `Math.nextUp(topScore)`, the next float above the 10th-best: documents arrive in docID order and ties break toward lower docIDs, so a later document with an equal score could never displace the incumbent, and the scorer may as well be told to require strictly more.

The bar changes the question the engine asks. It no longer needs each document's score, only **whether a document could clear the bar**, and "could" is answerable with an upper bound that §5 hands out for free. Note the direction of control: the collector never skips anything itself. It publishes a number, and the scorer does the skipping.

**What the threshold counts.** `totalHits` goes up by one per call to `collect()`: once per document that matched and was scored. Not documents examined, since skipped blocks never reach the collector, and not per segment either — it is one counter for the whole slice, so our twelve segments share it. Past the threshold the counter stops meaning "how many matched", because the scorer has stopped offering the collector documents it can prove are hopeless. That is the same fact as §6's hit count going fuzzy, seen from the other end.

**And why 1,000?** Not for speed. A lower threshold would be *faster*: set it to `numHits` and the bar would arm after eleven hits, pruning would start almost at once, and the results would be identical. The number buys an accurate hit count, nothing else, and 1,000 is where that stops being cheap to buy. Under a thousand matches you get an exact count and lose nothing, because scoring a thousand documents was never expensive. Over a thousand, you are in the case where pruning actually pays and where no interface renders the exact figure anyway. The threshold also has a floor of `numHits` (`Math.max(totalHitsThreshold, numHits)`), which is what makes deep paging expensive: asking for the top 5,000 forces 5,000 documents to be scored the slow way before any bar exists.

Say the search has run a while and the bar sits at **7.2**.

### Skip 1: skipping whole blocks

Before decoding any block of postings, the engine reads its signpost, a few bytes, and compares the block's ceiling against the bar. Walking `dog`'s list with the bar at 7.2:

| block | signpost says best possible score is… | bar | what happens |
|---|---|---|---|
| block 8 | 5.9 | 7.2 | **skipped** — 256 documents never decoded |
| block 9 | 4.1 | 7.2 | **skipped** — 256 documents never decoded |
| block 10 | 8.8 | 7.2 | decoded, its documents scored one by one |
| block 11 | 6.0 | 7.2 | **skipped** |

512 documents disappeared for the cost of reading two signposts. Nothing was approximated: a block whose *best imaginable* document scores 5.9 genuinely cannot contain a member of a top 10 whose worst member scores 7.2.

That's the whole mechanism, and it has a name. The stored ceilings are a **block-max index** (Ding & Suel, 2011); the family of algorithms that consume them are called **MAXSCORE** and **WAND**. Lucene implements both and picks between them per query shape — a bare term query, a pure `OR`, an `OR` with a "must match at least *k*" constraint, and a mixed required/optional query each get a different scorer class. The differences are mechanical rather than conceptual: same stored bounds, same exactness guarantee, different bookkeeping for tracking which clauses still matter. If you need the specifics, the choice is made in `search/BooleanScorerSupplier.java`, and the implementations are `MaxScoreBulkScorer`, `WANDScorer` and `ImpactsDISI`.

Skip 2 below describes the idea our example query actually uses.

### Skip 2: skipping a whole clause

Skip 1 works inside one posting list. Skip 2 asks a bigger question about the stretch of index ahead: **does this clause need to drive the walk at all, or can it just answer questions?**

With the bar at 7.2, here are the next few documents the engine would meet if it walked both lists naively:

| docID | contains | `quick` contributes | `dog` contributes | total | vs. bar 7.2 |
|---|---|---|---|---|---|
| 41,001 | `quick` only | 4.8 | — | **4.8** | loses |
| 41,233 | `dog` only | — | 8.1 | **8.1** | wins — enters the leaderboard |
| 41,502 | both | 3.9 | 6.7 | **10.6** | wins |
| 41,780 | `quick` only | 5.5 | — | **5.5** | loses |
| 41,905 | `quick` only | 3.1 | — | **3.1** | loses |

Look at which rows won: **every one of them contains `dog`.** And that isn't luck about these five documents — it's guaranteed by the ceiling table above. A document matching `quick` alone can score at most 5.9 in the stretch of the index the engine is currently working through, and 5.9 is below the bar. No `quick`-only document *in this stretch* can reach the top 10.

A posting list does two different jobs:

1. **It produces candidates** — "here is the next document that contains me."
2. **It contributes score** — "for the document you're asking about, here's what I'm worth."

We've just proved that `quick` can never produce a *winning* candidate on its own. But we still need job 2: document 41,502 matches both terms, and without `quick`'s 3.9 its score would be wrong. So `quick` isn't dropped — it's **demoted**. It stops producing candidates and answers questions instead:

```
  walk dog's list:      41,233        41,502        42,110        ...
                          │             │             │
  ask quick:        "have 41,233?"  "have 41,502?"  "have 42,110?"
                        no            yes → +3.9      no
                          │             │             │
  score:                 8.1          10.6           7.9

  meanwhile:  41,001, 41,780, 41,905 and ~11,600 other quick-only documents
              are never visited, never decoded, never scored
```

`quick` has 12,000 documents; only about 400 of them also contain `dog`. So demoting `quick` removes roughly 11,600 documents from the walk — and each remaining question ("do you contain 41,233?") is answered with an `advance()` that uses the signposts from §5, so it's a couple of block-boundary reads rather than a scan.

**The rule that decides this** generalizes: a clause can be demoted to lookup-only as long as the ceilings of **all** demoted clauses, added together, stay below the bar. Here only `quick` is demoted, and 5.9 < 7.2, so the split is legal. Add a third clause `labrador` with a ceiling of 2.0 and the two could be demoted together only while 5.9 + 2.0 = 7.9 stays below the bar — because a document matching both would score up to 7.9.

Lucene's names for the two groups are **essential** (walked) and **non-essential** (looked up).

**The split is per window, not per query** — which is what lets the scheme work without a whole-list ceiling. `MaxScoreBulkScorer` walks the segment in *outer windows*, each ending at the next block or region boundary among the clauses currently doing the walking. For every window it asks each clause for `getMaxScore(windowEnd)`, a bound valid only inside that window, and redoes the partition. A clause demoted in one window can come back as essential in the next, if that window happens to hold its high-scoring documents. So the split tracks the rising bar and the local shape of the data at once:

```
  bar = 0.0  (fewer than 1,000 hits collected — no bar published yet)
      dog   (3,000 docs)  ──┐
                            ├──▶ both lists walked and merged   (~14,600 documents)
      quick (12,000 docs) ──┘

  bar = 7.2  (above quick's ceiling of 5.9)
      dog   (3,000 docs)  ──────▶ walked                        ← the only driver
      quick (12,000 docs) ─ ─ ─ ▶ looked up at dog's 3,000 docIDs
```

None of this is an approximation. Every skip is justified by a ceiling provably below the bar, so you get the same ten documents, in the same order, as scoring all 14,600 would.

But it does need a bar, and the bar arrives late. The collector publishes its 10th-best score only after it has collected more than `totalHitsThreshold` hits, 1,000 by default, so until then nothing can be skipped. That's why the first thousand or so hits are scored the boring way, and why the technique pays off precisely on the queries that match a lot of documents.

> **Under the hood.** `MaxScoreBulkScorer#partitionScorers` implements the demotion test, greedily filling the non-essential set from a list sorted by `maxWindowScore / cost` — a refinement on textbook MAXSCORE, which sorts on max score alone. The two orders agree whenever max score is inversely correlated with document frequency, which is the usual case; they diverge under heavy query-time boosts or custom scores. The float arithmetic is guarded too: sums of more than two max scores go through `MathUtil.sumUpperBound`, which inflates them by their worst-case accumulated error, so rounding can only ever make the scorer too cautious.

### The catch: you no longer know how many results there were

Every search result page has a line like *"about 14,600 results"*. In Lucene that number is `TopDocs.totalHits`, and it comes back from the same `search()` call as the ten documents.

Run our query and you don't get 14,600. You get something like **"1000+"** — `totalHits` is a pair of *(value, relation)*, and the relation has flipped from `EQUAL_TO` to `GREATER_THAN_OR_EQUAL_TO`. The value is however many documents Lucene actually visited, which past the threshold is no longer the same as how many matched.

This is not a shortcut Lucene takes to save a little time. It is the price of §6, and the two are the same knob:

```java
if (totalHits > totalHitsThreshold) {      // 1,000 by default
    scorer.setMinCompetitiveScore(...);    // <- the bar goes up, skipping begins
    totalHitsRelation = GREATER_THAN_OR_EQUAL_TO;   // <- and the count goes fuzzy
}
```

Those two lines are next to each other in `TopScoreDocCollector` because they are two consequences of one decision. To know that exactly 14,600 documents match, you must *visit* 14,600 documents. But §6's entire method is refusing to visit documents it can prove won't win. The moment Lucene starts skipping blocks, it loses count — and it would rather skip and admit the count is a lower bound than visit everything to produce a number most users read as "lots".

You can demand the exact figure by setting `totalHitsThreshold` to `Integer.MAX_VALUE`. Be clear about what that buys: the collector then reports `ScoreMode.COMPLETE` instead of `TOP_SCORES`, and Lucene builds a **different scorer** — one that never consults a score bound. Not "the same query, a bit slower": every block decoded, every match scored, an order of magnitude more work on a common query.

If you genuinely need a count, ask for a count rather than for documents. `IndexSearcher.count(query)` is free to use tricks a ranked search cannot, because it never has to produce a document. Counting a single term is the extreme case: on a segment with no deletions, `TermQuery`'s count is just `docFreq` read out of the term dictionary — the number §4 already had before touching a posting list. Nothing is walked at all.

It also knows the inclusion–exclusion identity for a two-term disjunction:

```
   |quick OR dog|  =  |quick| + |dog| - |quick AND dog|
```

with the first two terms free and only the intersection needing work. Lucene applies this one narrowly: the segment must have no deletions, and the two terms must be lopsided (`min/max < 0.1`), since otherwise the intersection costs about as much as counting the union directly. Our `quick` and `dog` sit at 3,000/12,000 = 0.25, so this particular query wouldn't qualify. The general point stands: `count()` gets to pick a strategy that `search()` can't.

And if the number is only ever going to be rendered as "about 14,000 results", take the lower bound and round it. That's what it's for.

### Sorting by a field instead

Everything above ranks by score. If you sort by `rating` instead — `search(q, 10, new Sort(new SortField("rating", LONG, true)))` — the shape of the argument survives but every part of the machinery is replaced.

| | ranking by relevance | sorting by `rating` |
|---|---|---|
| collector | `TopScoreDocCollector` | `TopFieldCollector` |
| the bar | 10th-best **score**, published with `setMinCompetitiveScore` | 10th-best **rating**, published with `comparator.setBottom(...)` |
| where the bound is read from | impacts in `.doc` | the BKD point tree, or a doc-values skip index |
| who does the skipping | `MaxScoreBulkScorer` / `ImpactsDISI` | `competitiveIterator()`, from `NumericComparator` |

Norms, IDF and impacts play no part; with `doDocScores` off, scores may never be computed at all. The idea is identical — keep a bar, get a cheap upper bound, skip what cannot clear it — but the two paths share no structures, and none of §5's block-max data is touched.

One trap. Doc values are enough to *read* a field's value, not to skip on it:

```java
PointValues pointValues = reader.getPointValues(field);
if (pointValues != null) return new PointsCompetitiveDISIBuilder(pointValues, this);
DocValuesSkipper skipper = reader.getDocValuesSkipper(field);
if (skipper != null) return new DVSkipperCompetitiveDISIBuilder(skipper, this);
return null;   // no pruning
```

A field with doc values and nothing else falls through to `null`, and every match gets visited. To sort on a numeric field cheaply, index it as a point as well, or write its doc values with a skip index. That is the same advice as §8's `IndexOrDocValuesQuery` note, arrived at from the sorting side rather than the filtering side.

### AND queries plan differently

For `+quick +dog` (both required), no bar is needed to be clever. The engine knows each clause's size from §4, 12,000 and 3,000, so it walks the **smaller** list and advances the larger one to those docIDs. A conjunction costs roughly what its rarest term costs, which is why adding a rare required term to a query usually makes it *faster*, not slower.

The same "cheap test first" principle covers expensive predicates. A phrase check, a geo-distance test, or a script filter is split into a cheap **approximation** (for the phrase `"quick brown"`: documents containing both words) that joins the intersection, and an expensive confirmation that runs only on documents that already survived everything else. That's why the phrase query reads positions for a few hundred documents instead of 12,000.

---

## 7. Steps 7-8: Merging, fetching, and adding it all up

Everything so far happened inside one segment. Two steps remain.

### Step 7: merging the per-slice results

Each slice has produced a top 10, scored with the *same* index-wide BM25 weights (that's why step 1 gathered statistics across all segments before any of them was touched — otherwise the same document would rank differently depending on which segment it landed in). Merging is therefore an n-way merge of sorted lists, keeping the best 10 overall. How big n is depends on concurrency, not on segment count: single-threaded, there is one list and the "merge" is a formality; with an executor, one list per slice.

The one fiddly part is identity: each segment numbered its documents from 0, so segment `_7`'s document 41,233 and segment `_8`'s document 41,233 are different documents. The translation happens earlier than you might expect. Not in the merge, but at collection time: `TopScoreDocCollector` stores `doc + docBase` (where `docBase` is the number of documents in all preceding segments) the moment a hit enters the queue. This is also what lets one leaderboard span several segments safely. By the time `TopDocs.merge` runs, every docID is already index-wide and the merge only has to compare scores.

### Step 8: fetching the fields to display

Ranking used only the index structures: posting lists, norms, live docs. Nobody has yet read the actual text of a review. That happens now, and only for the 10 documents being returned:

```java
StoredFields fields = searcher.storedFields();
for (ScoreDoc hit : top.scoreDocs) {
    Document doc = fields.document(hit.doc, Set.of("title"));   // 10 documents, 1 field
}
```

Stored fields are compressed in chunks spanning many documents, so each fetch decompresses a chunk. Ten fetches is nothing; ten thousand would dominate the query. Hence the rule: never touch stored fields during matching, only for the page you're about to render.

### Adding it all up: what the query actually read

Every section of this article described one mechanism for avoiding work. This table puts them side by side so you can see the combined effect — it's the summary of the whole article, one row per mechanism.

The comparison is between a **straightforward implementation** (find all matches, score them, sort) and what Lucene actually does, for our one query (`quick OR dog`, top 10) in segment `_7`: 524,288 documents, about 14,600 of which match.

| stage | a straightforward engine reads… | Lucene reads… | § |
|---|---|---|---|
| find the terms | 2 dictionary lookups per segment, one after another | trie walks in mapped memory; absent terms rejected with **zero** disk reads; the rest issue their I/O in parallel across all 12 segments | §4 |
| read docIDs | all ~14 KB of posting data | all of `dog` (3.3 KB); of `quick`, only the blocks holding a candidate — and once `quick` is demoted to lookups, whole regions are bounded out unread | §5, §6 |
| read frequencies | another ~4 KB | only for blocks that actually produce a candidate | §5 |
| read positions | both terms' `.pos` files | **zero bytes** — this query has no phrase constraint | §5 |
| score documents | all 14,600 | ~1,000, until the hit threshold is crossed and the bar appears; after that, only `dog`'s still-competitive candidates | §6 |
| read display text | often, for every match | 10 documents, after ranking, `title` only | §7 |
| across the index | repeat per segment, then sort everything | the same per-segment work ×12, run in parallel, merged into one global top 10, each segment pruning against its own bar | §2, §7 |

Roughly: ~18 KB of posting data decoded and 14,600 documents scored, against a few KB decoded and a few thousand scored, for **identical results** in the same order.

Every row is the same move. The engine spends a few bytes to learn a bound, then uses the bound to avoid kilobytes of work.

---

## 8. A search performance checklist

Concrete things that move query latency on an index like our 2-million-document example, roughly in order of impact:

1. **Watch segment count.** Every query is executed once per segment. Let the merge policy do its job; for a static index, a `forceMerge(1)` before going read-only is a real win. Don't force-merge an index you're still writing to.
2. **Don't let deletions pile up.** Deleted documents are read out of posting lists and then discarded. Past ~20% deleted, you're paying real I/O for documents nobody will ever see.
3. **Don't ask for exact hit counts.** Leaving `totalHitsThreshold` at its default of 1,000 keeps the collector in `TOP_SCORES` mode, which is what makes §6 possible at all. Setting it to `Integer.MAX_VALUE` switches to `COMPLETE`, builds a scorer that ignores score bounds, and can cost an order of magnitude on a common query. Note the default is a hit-count knob, not a speed knob: a *lower* threshold would prune sooner, at the price of a less useful count.
4. **Fetch stored fields last, and only for the page.** Ten documents, only the fields you'll render. Never inside a collector.
5. **Leave RAM for the page cache.** Index data lives in mapped files; the OS caches it. A huge heap actively hurts. Warm new readers after a reopen if first-query latency matters.
6. **Use filter clauses for anything not contributing to relevance** (`BooleanClause.Occur.FILTER`, `ConstantScoreQuery`). They skip scoring entirely and become eligible for the query cache, which stores them as bitsets. Note that caching is **off** by default in this version of Lucene (`IndexSearcher`'s default `QueryCache` is `null`); install one with `IndexSearcher.setQueryCache(new LRUQueryCache(...))` if repeated filters are a large part of your load.
7. **Reuse readers via `SearcherManager`.** Don't open a `DirectoryReader` per request; do reopen periodically rather than never.
8. **Avoid deep paging.** `totalHitsThreshold` has a floor of `numHits`, so asking for the top 5,000 to reach page 500 scores 5,000 documents before any pruning can start. `searchAfter` keeps `numHits` at one page, so the leaderboard and the threshold stay small.
9. **For range fields, index both a point and a doc-values field** and wrap them in `IndexOrDocValuesQuery`, which picks per segment between "look it up in the index" and "scan the column" based on cost.
10. **Measure before tuning.** `IndexSearcher.explain()` tells you why a document scored what it did; `CheckIndex -verbose` tells you what your term dictionary and segments actually look like.

---

## 9. Why the design looks like this

Part 1's write path is organized around **avoiding coordination**: private per-thread buffers, immutable output, publication at the last possible moment. The read path solves a different problem. The index is far larger than RAM and a query has milliseconds, so it is organized around **avoiding bytes**. Six principles do the work below, and every one of them is a trade with a real cost.

### 1. Immutability is the foundation, and it is not free

Every good property in this article descends from one decision: a segment's bytes never change.

| what immutability buys | what it costs |
|---|---|
| readers need **no locks** — a mapped file that can't change is safe to read forever | you can't update a document, only add a new one and tombstone the old |
| a reader is a consistent **snapshot** for free, with no MVCC machinery | deleted documents keep occupying posting lists until a merge rewrites them |
| aggressive **caching** — a segment's data and any derived structure stay valid for that segment's whole life | docIDs are not stable identifiers; they change on every merge |
| the OS page cache does the caching, so there's **no application-level buffer pool** | merging is a permanent background cost paid in I/O and CPU |

None of those costs are avoided; they're deferred. Merging is where the bill comes due, and the whole bargain is to make reads cheap and pay later in a background process you get to schedule.

### 2. Metadata eagerly, data lazily — at six levels

Laziness here is not one decision but the same decision taken six times over. Each level is a chance to stop before touching the next:

```
   open a reader        -> read metadata files only, map the rest       (§3)
   look up a term       -> walk the term index; often prove absence     (§4)
                           without reading the dictionary at all
   read a block         -> only blocks the signposts say could matter   (§5, §6)
   decode frequencies   -> only when a document is actually scored      (§5)
   read positions       -> only when a phrase check demands them        (§5)
   read stored fields   -> only for the documents being returned        (§7)
```

Our query descended four of those levels and never reached the last two: no positions were read at all, and stored fields were touched for ten documents. A phrase query on a rare term would descend all six, but on far fewer documents. The structure is what makes both cheap.

### 3. Exact statistics, available before the work starts

Query planners in databases usually estimate cardinality from sampled histograms, and they estimate *once*, before execution. Lucene is in an unusually lucky position. **`docFreq` is exact, per segment, and free:** it's written next to the term, so it arrives with the lookup, before any posting data is read.

That single property is why the read path can afford to plan so aggressively:

- Plans are made **per segment**, not once per query. `dog` may be the rare term in `_7` and the common one in `_9`; each segment gets its own decision.
- Plans are revised **mid-execution**, and often. The essential/non-essential split in §6 is recomputed for every window the scorer enters — driven both by the rising bar and by the local impacts of the window ahead — which is only affordable because the inputs are a few bytes of already-mapped skip data.
- The `ScorerSupplier` layer exists to hold open the moment where cost is known but no work has been committed. That is the planner's decision point, and the reason the class exists at all.

### 4. Bounds, not heuristics — which constrains what scoring can be

The skipping in §6 is sound only because a score can be bounded before it is computed, and that is possible only because BM25 is **monotonic**: more occurrences can only raise a score, a longer field can only lower it. That lets the index store a per-region ceiling and lets the engine reason "nothing in here can beat the bar."

That is a real constraint on what you can do with Lucene, and it bites before you write any code:

- Custom similarities are expected to preserve monotonicity. One that doesn't makes the stored ceilings wrong, and wrong ceilings mean wrong results, not merely slow ones.
- A score that Lucene cannot bound — an arbitrary script, a function over doc values, a late-stage rescoring model — reports an infinite ceiling, which disables pruning. The query silently falls back to visiting every match. It still works; it just costs what §7's "straightforward engine" column costs.
- The usual remedy is two-stage: let a boundable query (BM25) produce a few hundred candidates cheaply, then apply the expensive unboundable model only to those. That's the same "cheap approximation, expensive confirmation" pattern as the phrase check in §6, one level up.

### 5. One sort order, exploited everywhere

Almost every structure in a segment is sorted by docID, and almost every optimization in this article is a consequence of that one choice:

| because docIDs are sorted and ascending… | we get |
|---|---|
| differences between them are small | gap encoding, and blocks that pack into 7 bits (§5) |
| a list can be scanned in one direction | skip signposts, and `advance()` that never backtracks (§5) |
| two lists can be walked together | intersection as a merge join — no hash tables, no materialization (§6) |
| norms and live docs share the ordering | scoring reads them sequentially alongside the postings (§6) |
| a dense list is just a bitmap | the bitset encoding that makes `the` cheap (§5) |

Choosing a single canonical order early and then refusing to deviate from it pays off in places you would never have listed up front. Six of them are in the table above.

### 6. Batch everything, at a size chosen to fit the machine

Nothing here is processed one item at a time. Postings decode 256 at a time so the unpacking loop vectorizes; scoring runs over windows of a few thousand documents into flat arrays instead of virtual calls per document; stored fields compress in chunks; term dictionary entries group into blocks by shared prefix.

The block size is a compromise you can now read off the mechanisms: **larger blocks** decode more efficiently per document and compress better, **smaller blocks** give finer skip granularity and waste less when you only want a few documents from a region. 256 is where those curves crossed for this codec on current hardware, which is why the number has changed across Lucene versions and will change again.

### What it adds up to

The read path rarely asks how to do a piece of work faster. It asks how to prove the work is unnecessary, then makes sure the proof costs less than the work would have. Metadata is read so data can be skipped, ceilings are read so scores can be skipped, and at every level the engine spends a few bytes to avoid kilobytes.

Stated as a design rule: **push cheap, exact summaries as close to the data as you can, and make every expensive operation ask a summary for permission first.** Lucene can do this because `docFreq` and the impacts are exact rather than sampled. A system whose summaries are estimates has to handle being wrong, which is a different and much harder design.

*Next in this series: what happens when segments merge — how Lucene rewrites and consolidates the very files this article read from, why the read path's costs make merging unavoidable, and what it costs a live system.*

---

## 10. Caveats and version notes

- **The example numbers are constructed, not measured.** Document counts, term frequencies, block counts and score bounds are internally consistent arithmetic over the real format constants (256-entry postings blocks, coarse signposts every 32 blocks, a default hit threshold of 1,000, BM25's IDF formula), chosen to make mechanisms visible. Your data will differ; measure it.
- **Codec details move between releases.** This article describes a recent Lucene: a trie-based term index and 256-document postings blocks. Lucene 10.x releases in the wild use an FST-based term index and 128-document blocks. The *architecture* — two-level term lookup, block postings, two-level skips, impacts — has been stable for years; the constants and class names have not.
- **Simplified deliberately:** the query cache (`LRUQueryCache` — off by default here, but a large effect for repeated non-scoring clauses once enabled); the concurrency model beyond "segments and segment slices are searched in parallel"; numeric/geo search over BKD trees; and vector/KNN search. Each deserves its own article.
- **The source tree this was written against** is a working checkout that is slightly ahead of, and in places diverges from, released Apache Lucene. Where you plan to rely on a specific class name or constant, check it against the version you actually run.

---

## 11. Appendix: references and code artifacts

**Source paths** (relative to `lucene/core/src/java/org/apache/lucene/`), if you want to read along:

| Topic | Where to look |
|---|---|
| Opening and reopening readers | `index/DirectoryReader.java`, `index/StandardDirectoryReader.java`, `index/SegmentCoreReaders.java` |
| Memory-mapped I/O | `store/MMapDirectory.java`, `store/MemorySegmentIndexInput.java`, `store/ReadAdvice.java` |
| Term dictionary and term index | `codecs/lucene103/blocktree/` (`Lucene103BlockTreeTermsReader`, `SegmentTermsEnum`, `TrieReader`) |
| Posting lists, skips, impacts | `codecs/lucene104/Lucene104PostingsFormat.java` (the format is documented in the class javadoc), `Lucene104PostingsReader.java` |
| Query execution and planning | `search/IndexSearcher.java`, `search/Weight.java`, `search/ScorerSupplier.java`, `search/TermQuery.java` |
| Dynamic pruning | `search/ImpactsDISI.java`, `search/MaxScoreBulkScorer.java`, `search/WANDScorer.java`, `search/TopScoreDocCollector.java` |
| Scoring | `search/similarities/BM25Similarity.java` |
| Near-real-time serving | `search/SearcherManager.java`, `search/ControlledRealTimeReopenThread.java` |

**Further reading**

- Apache Lucene documentation and API: <https://lucene.apache.org/core/>
- Source: <https://github.com/apache/lucene>
- Turtle & Flynn, *Query Evaluation: Strategies and Optimizations* (IP&M 1995) — the original MAXSCORE algorithm, the one §6's essential/non-essential split implements.
- Broder et al., *Efficient Query Evaluation using a Two-Level Retrieval Process* (CIKM 2003) — the WAND algorithm, Lucene's `WANDScorer`.
- Ding & Suel, *Faster Top-k Document Retrieval Using Block-Max Indexes* (SIGIR 2011) — the impacts idea in §5–6.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).

**Series**

- Part 1 — [Deep Dive into Apache Lucene's Core: From In-Memory Buffers to Commits](https://kurtzhao.com/posts/deep-dive-into-apache-lucene-core/)
- Part 2 — Deep Dive into Apache Lucene's Search Path *(this article)*
