---
layout: post
title: Understanding Conflict-Free Replicated Data Types (CRDTs)
date: 2025-05-04 14:46 -0400
---
# Introduction

Modern distributed systems often replicate data across nodes for availability and low latency. However, concurrent updates on different replicas without coordination can lead to conflicts and inconsistencies. Traditional strong-consistency approaches (locking, consensus) impose high coordination costs and latency. CRDTs (Conflict-Free Replicated Data Types) were introduced to solve this problem: they are specially designed data structures that allow each replica to accept updates independently and then merge divergent states deterministically, ensuring all replicas eventually converge to the same state.

In other words, CRDTs enable optimistic replication: each replica proceeds with local updates and later reconciles with others. As Shapiro et al. note, replicas of a CRDT “are guaranteed to converge in a self-stabilising manner, despite any number of failures”.

When two replicas update shared data independently (e.g. in different geographic regions), their states may temporarily diverge. CRDTs define merge functions that are commutative, associative, and idempotent, so merging out-of-order or duplicate messages does not change the final result. In CRDTs, “the application can update any replica independently, concurrently and without coordinating,” and an embedded merge algorithm “automatically resolves any inconsistencies,” guaranteeing eventual convergence.

In practice, CRDTs provide strong eventual consistency: as long as no new updates arrive and replicas continue exchanging (merged) states, they will all end up with identical data. This makes CRDTs ideal for highly available, geo-distributed apps (collaborative editors, caches, IoT data, etc.) where network partitions or latencies are common.


# What is CRDT 
CRDT: Conflict-free replicated data type  is a data structure that is replicated across multiple computers in a network, with the following features:
- The application can update any replica independently, concurrently and without coordinating with other replicas. 
- An algorithm (itself part of the data type) automatically resolves any inconsistencies that might occur 
- Although replicas may have different state at any particular point in time, they're guaranteed to eventually converge. 

# How CRDT Works 

There are two approaches to CRDTs, both of which can provide strong eventual consistency: 
- state-based CRDTs 
- operation-based CRDTs.

## State-based CRDTs

State-based CRDT is also called convergent replicated data types, or CvRDTs

It is defined by two types:
- a type for local states 
- a type for actions on the state , together with three functions
	- produce an initial state 
	- a merge function of states 
	- a function to apply an action to update a state 

State-based CRDTs simply send their full local state to other replicas on every update, where the received new state is then merged into the local state.

To ensure eventual convergence, merge function must be:
- commutative: (merge(a,b) = merge(b,a)) – order of merging doesn’t matter.
- associative:  (merge(a, merge(b,c)) = merge(merge(a,b),c)) – grouping of merges doesn’t matter.
- and idempotent.  : (merge(a,a) = a) – merging the same state twice has no effect.

The intuition behind commutativity, associativity and idempotence is that these properties are used to make the CRDT invariant under package re-ordering and duplication. 

Delta state CRDTs (or simply Delta CRDTs) are optimized state-based CRDTs where only recently applied changes to a state are disseminated instead of the entire state.

## Operation-based CRDTs

Operation-based CRDTs is also called commutative replicated data types, or CmRDTs are defined without a merge function. 

Instead of transmitting states, the update actions are transmitted directly to replicas and applied. For example, an operation-based CRDT of a single integer might broadcast the operations (+10) or (−20). 

The application of operations should still be 
- commutative and 
- associative. 
- However, instead of requiring that application of operations is idempotent, stronger assumptions on the communications infrastructure are expected -- all operations must be delivered to the other replicas without duplication.

The two alternatives are theoretically equivalent, as each can emulate the other.[1] However, there are practical differences. State-based CRDTs are often simpler to design and to implement; their only requirement from the communication substrate is some kind of gossip protocol. 

Their drawback is that the entire state of every CRDT must be transmitted eventually to every other replica, which may be costly.

In contrast, operation-based CRDTs transmit only the update operations, which are typically small. However, operation-based CRDTs require guarantees from the communication middleware; that the operations are not dropped or duplicated when transmitted to the other replicas, and that they are delivered in causal order

Gossip protocols work well for propagating state-based CRDT state to other replicas while reducing network use and handling topology changes.

# Core-properties

Commutative, associative and idempotent ensure state convergence is independent of message ordering or repetition. Moreover, every update to the state must be monotonic: it produces a new state that is greater or equal (new ≥ old) in the partial order. Intuitively, this means updates only add information (never “undo” or shrink the state). For instance, adding an element to a grow-only set never makes it smaller. This monotonicity, combined with the ACI merge, guarantees strong eventual consistency: as long as all updates eventually propagate and merge, replicas converge。


Marc Shapiro et al. summarize the requirements succinctly: “In the state-based style, the successive states of an object should form a monotonic semilattice, with merge computing a least upper bound. In the op-based style, concurrent operations should commute.”

In other words, either we make state merges form a join-semilattice (CvRDT) or we make the operations themselves form a commutative monoid (CmRDT). If these conditions hold (and the network eventually delivers all messages), the system achieves Strong Eventual Consistency: no matter how replicas diverge, merging automatically yields a common, correct state without needing any central coordination.

# CRDT Implementation:

A state-based CRDT is any data structure that implements this interface:

```
interface CRDT<T, S> {
  value: T;
  state: S;
  merge(state: S): void;
}
```

A value, T. This is the part the rest of our program cares about. The entire point of the CRDT is to reliably sync the value between peers.


state: A state, S. This is the metadata needed for peers to agree on the same value. To update other peers, the whole state is serialized and sent to them.

A merge function. This is a function that takes some state (probably received from another peer) and merges it with the local state.

The merge function must satisfy three properties to ensure that all peers arrive at the same result 
 - Commutativity : states can be merged in any order; A ∨ B = B ∨ A. If Alice and Bob exchange states, they can each merge the other’s state into their own and arrive at the same result . -> order of merging doesn’t matter.
 - Associativity:  when merging three (or more) states, it doesn’t matter which are merged first; (A ∨ B) ∨ C = A ∨ (B ∨ C). If Alice receives states from both Bob and Carol, she can merge them into her own state in any order and the result will be the same.  -> grouping of merges doesn’t matter.
 - Idempotence: merging a state with itself doesn’t change the state; A ∨ A = A. If Alice merges her own state with itself, the result will be the same state she started with. -> merging the same state twice has no effect.

## Common Known CRDTs

### G-Counter (Grow-only Counter)

Pseudo code:

```
payload integer[n] P
    initial [0,0,...,0]
update increment()
    let g = myId()
    P[g] := P[g] + 1
query value() : integer v
    let v = Σi P[i]
compare (X, Y) : boolean b
    let b = (∀i ∈ [0, n - 1] : X.P[i] ≤ Y.P[i])
merge (X, Y) : payload Z
    let ∀i ∈ [0, n - 1] : Z.P[i] = max(X.P[i], Y.P[i])
```


python code:

```python
class GCounter:
    def __init__(self, node_id):
        self.node_id = node_id
        self.counts = defaultdict(int)  # map from node_id -> count

    def increment(self, by=1):
        # Only increase own count
        self.counts[self.node_id] += by

    def value(self):
        # The total count is sum of all entries
        return sum(self.counts.values())

    def merge(self, other):
        # Merge by taking max for each node
        for nid, cnt in other.counts.items():
            self.counts[nid] = max(self.counts.get(nid, 0), cnt)
```

This state-based CRDT implements a counter for a cluster of n nodes. Each node in the cluster is assigned an ID from 0 to n - 1, which is retrieved with a call to myId(). Thus each node is assigned its own slot in the array P, which it increments locally. Updates are propagated in the background, and merged by taking the max() of every element in P. The compare function is included to illustrate a partial order on the states. The merge function is commutative, associative, and idempotent. The update function monotonically increases the internal state according to the compare function. This is thus a correctly defined state-based CRDT and will provide strong eventual consistency. The operations-based CRDT equivalent broadcasts increment operations as they are received.

## PN-Counter (Positive-Negative Counter)

By combining two G-Counters, we get a counter that supports increments and decrements. Each replica maintains one G-Counter for “increments” and another for “decrements”. The value is sum(P) – sum(N). 

Pseudocode:


```
payload integer[n] P, integer[n] N
    initial [0,0,...,0], [0,0,...,0]
update increment()
    let g = myId()
    P[g] := P[g] + 1
update decrement()
    let g = myId()
    N[g] := N[g] + 1
query value() : integer v
    let v = Σi P[i] - Σi N[i]
compare (X, Y) : boolean b
    let b = (∀i ∈ [0, n - 1] : X.P[i] ≤ Y.P[i] ∧ ∀i ∈ [0, n - 1] : X.N[i] ≤ Y.N[i])
merge (X, Y) : payload Z
    let ∀i ∈ [0, n - 1] : Z.P[i] = max(X.P[i], Y.P[i])
    let ∀i ∈ [0, n - 1] : Z.N[i] = max(X.N[i], Y.N[i])
```

python

```python
class PNCounter:
    def __init__(self, node_id):
        self.node_id = node_id
        self.P = defaultdict(int)  # increments per node
        self.N = defaultdict(int)  # decrements per node

    def increment(self, by=1):
        self.P[self.node_id] += by

    def decrement(self, by=1):
        self.N[self.node_id] += by

    def value(self):
        # net count is increments minus decrements
        return sum(self.P.values()) - sum(self.N.values())

    def merge(self, other):
        # Merge each sub-counter independently
        for nid, cnt in other.P.items():
            self.P[nid] = max(self.P.get(nid, 0), cnt)
        for nid, cnt in other.N.items():
            self.N[nid] = max(self.N.get(nid, 0), cnt)
```


A common strategy in CRDT development is to combine multiple CRDTs to make a more complex CRDT. In this case, two G-Counters are combined to create a data type supporting both increment and decrement operations. The "P" G-Counter counts increments; and the "N" G-Counter counts decrements. The value of the PN-Counter is the value of the P counter minus the value of the N counter. Merge is handled by letting the merged P counter be the merge of the two P G-Counters, and similarly for N counters. Note that the CRDT's internal state must increase monotonically, even though its external state as exposed through query can return to previous values.

## Last Write Wins Register

A register is a CRDT that holds a single value. There are a couple kinds of registers, but the simplest is the Last Write Wins Register (or LWW Register).

Here’s the algorithm:

- If the received timestamp is less than the local timestamp, the register doesn’t change its state.
- If the received timestamp is greater than the local timestamp, the register overwrites its local value with the received value. It also stores the received timestamp and some sort of identifier unique to the peer that last wrote the value (the peer ID).
- Ties are broken by comparing the local peer ID to the peer ID in the received state.

```java
class LWWRegister<T> {
  readonly id: string;
  state: [peer: string, timestamp: number, value: T];

  get value() {
    return this.state[2];
  }

  constructor(id: string, state: [string, number, T]) {
    this.id = id;
    this.state = state;
  }

  set(value: T) {
    // set the peer ID to the local ID, increment the local timestamp by 1 and set the value
    this.state = [this.id, this.state[1] + 1, value];
  }

  merge(state: [peer: string, timestamp: number, value: T]) {
    const [remotePeer, remoteTimestamp] = state;
    const [localPeer, localTimestamp] = this.state;

    // if the local timestamp is greater than the remote timestamp, discard the incoming value
    if (localTimestamp > remoteTimestamp) return;

    // if the timestamps are the same but the local peer ID is greater than the remote peer ID, discard the incoming value
    if (localTimestamp === remoteTimestamp && localPeer > remotePeer) return;

    // otherwise, overwrite the local state with the remote state
    this.state = state;
  }
}
```

- state is a tuple of the peer ID that last wrote to the register, the timestamp of the last write and the value stored in the register.
- value is simply the last element of the state tuple.
- merge is a method that implements the algorithm described above.

LWW Registers have one more method named set, which is called locally to set the register’s value. It also updates the local metadata, recording the local peer ID as the last writer and incrementing the local timestamp by one.


# Real-World CRDT Systems

## Redis CRDT 

Redis use two stages to do CRDT:
- Stage 1. Classify the update operation based on vector clock. 
    - if: x_vc[b] > x_vc[a] – this is “new” update
    - if: x_vc[b] < x_vc[a] – this is an “old” update
    - if: x_vc[b] ≸ x_vc[a] – this is a “concurrent” update

- Stage 2. Update the object locally
    - If the update was classified as “new,” update the object value in the local CRDB instance
    - If the update was classified as “old,” do not update the object value in the local CRDB instance
    - If the update was classified as “concurrent,” perform conflict resolution to determine if and how to update the object value in this CRDB instance

The CRDB conflict resolution algorithm is based on two main processes:
- Process 1: Conflict resolution for a conflict-free data type/operation
	- *Counter*: All operations are commutative and conflict-free when the data type is Counter (mapped to a CRDT’s Counter). Example: Counters tracking article popularity, shares, or retweets across regions.
	- *Set and the concurrent updates are ADD operations*: All operations are either associative or idempotent and are conflict-free when the data type is Set (mapped to a CRDT’s Add-Wins Observed-Removed Set), and the concurrent updates are ADD operations. Example: In fraud detection, when the application is tracking dubious events associated with an ID or credit card. The Redis Set cardinality associated with the ID or credit card is used to trigger an alert if it reaches a threshold.
	- *Hash and the concurrent updates are over different Hash fields*: all operations are conflict-free. Example: A shared corporate account with multiple users, using a plan in different locations or at different rates. The HASH object per account could contain innumerable fields per user, and individual user usage could be updated conflict-free, even when concurrent. 

- Process 2: Conflict resolution using the Last Write Wins (LWW) mechanism
	- A conflict-resolution algorithm should be applied in cases of concurrent updates in a non-conflict-free data type, such as Redis String
	- We have used the LWW approach to resolve such situations by leveraging the operation timestamp as a tiebreaker.
	- Note that our solution works in a strong eventually consistent manner, even if there is a timestamp skew between regions. For example, assume Instance A’s timestamp is always ahead of the other instances’ timestamps  (i.e. in the case of a tiebreaker, Instance A always wins). This ensures behavior that is eventually consistent.


More information can be found at https://redis.io/active-active/.

# Reference 

- https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type
- https://redis.io/active-active/
- https://pages.lip6.fr/Marc.Shapiro/papers/RR-7687.pdf
- https://christophermeiklejohn.com/erlang/lasp/2019/03/08/monotonicity.html
- https://www.youtube.com/watch?v=B5NULPSiOGw
- https://martin.kleppmann.com/2020/07/06/crdt-hard-parts-hydra.html
- https://www.youtube.com/watch?v=GXJ0D2tfZCM
- https://github.com/automerge/automerge
- https://arxiv.org/pdf/1805.06358.pdf
- https://hal.inria.fr/inria-00555588/document
- https://martin.kleppmann.com/2019/03/25/papoc-interleaving-anomalies.html
- https://www.christopherbiscardi.com/crdt
- https://news.ycombinator.com/item?id=19341061
- https://gun.eco/distributed/matters.html
- https://youtu.be/yCcWpzY8dIA
- https://www.youtube.com/watch?v=oyUHd894w18






