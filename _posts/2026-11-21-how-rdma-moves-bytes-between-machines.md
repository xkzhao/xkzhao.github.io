---
layout: post
title: "How RDMA moves bytes between machines"
date: 2026-11-21 00:00:00 -0500
categories: [Systems, Networking]
tags: [rdma, networking, distributed-systems, roce, infiniband]
description: "Start with what a network adapter does. Then follow an animated remote write and see how storage and model-serving software use RDMA."
---

A program on server A wants to copy a record into a buffer on server B. A **buffer** is a range of memory set aside to hold bytes. With a regular network socket, B's program reads the incoming bytes into that buffer. With **Remote Direct Memory Access (RDMA)**, B can authorize A to write to a particular buffer, and B's network adapter can put the bytes there as they arrive.

To see why that matters, start with what a network adapter can do inside one computer.

<link rel="stylesheet" href="{{ '/assets/posts/rdma/figures.css' | relative_url }}">
<script defer src="{{ '/assets/posts/rdma/transfer.js' | relative_url }}"></script>

## A network adapter can move bytes on its own

A **network interface card (NIC)** is the hardware that sends and receives packets. It can be built into a machine or plugged in as a separate card. The **central processing unit (CPU)** runs your program and the operating system. **Random access memory (RAM)** holds working data, including the buffers your program uses.

When a device reads or writes local memory without the CPU copying the bytes itself, that's **Direct Memory Access (DMA)**. The software running on the CPU sets up the work. The device moves the bytes. Ordinary network adapters already use DMA. [Linux device memory guide][dma]

<figure class="rdma-figure">
  <figcaption>DMA inside one computer</figcaption>
  <div class="rdma-local">
    <div class="rdma-node rdma-control-node"><strong>CPU</strong><span>Runs the software</span></div>
    <div class="rdma-control-line">↓ Software sets up the transfer</div>
    <div class="rdma-local-path">
      <div class="rdma-node"><strong>Network adapter</strong><span>Moves the bytes</span></div>
      <div class="rdma-local-arrow"><span aria-hidden="true">↔</span><small>DMA</small></div>
      <div class="rdma-node"><strong>Memory</strong><span>Holds the buffer</span></div>
    </div>
  </div>
</figure>

## RDMA changes where incoming bytes land

On a typical **Transmission Control Protocol (TCP)** socket, the receiving adapter places packets in buffers managed by the operating system. The kernel, the core of the operating system, processes those packets. When the receiving program calls `recv()`, the kernel usually copies the requested bytes into the program's buffer. TCP gives the program a byte stream, so the program also has to work out where each record begins and ends. [TCP specification][tcp] · [Linux receive system calls][recv]

With an RDMA Write, the sending program names a destination in advance. The receiving adapter checks access and places the payload in that destination buffer. The kernel still helps set up the resources. The transfer can skip the usual per-message work in the kernel's network stack and the copy into the receiving program. [Linux userspace networking driver guide][uverbs] · [RDMA protocol specification][rdmap]

<figure class="rdma-figure">
  <figcaption>Where the bytes go on server B</figcaption>
  <div class="rdma-paths">
    <div class="rdma-path">
      <strong>Typical TCP socket</strong>
      <ol role="list">
        <li class="rdma-node">Network adapter</li>
        <li class="rdma-node">Operating system buffer</li>
        <li class="rdma-node">Program's buffer</li>
      </ol>
    </div>
    <div class="rdma-path rdma-direct">
      <strong>RDMA Write</strong>
      <ol role="list">
        <li class="rdma-node">Network adapter</li>
        <li class="rdma-direct-label">Places bytes directly</li>
        <li class="rdma-node">Program's buffer</li>
      </ol>
    </div>
  </div>
</figure>

## Follow one RDMA write

Say server A is copying a log record to server B. Each program **registers** its buffer with the RDMA driver before the transfer. Registration describes the memory range and its permitted uses to the adapter. B grants remote write access, then shares the destination address, its capacity, and a **remote key**, called `rkey`. The adapter uses the key to check the region's bounds and permissions. A also has a **local key**, called `lkey`, for its own buffer. [Linux memory registration manual][register]

A submits the operation as a **work request**. A **queue pair (QP)** has one queue for outgoing instructions and one for receive requests. A **completion queue (CQ)** holds results that the program can check. [RDMA programming manual][programming] · [Linux completion queue manual][poll]

<figure class="rdma-figure rdma-demo" data-rdma-demo>
  <figcaption>One record from A to B</figcaption>
  <div class="rdma-route" aria-hidden="true">
    <span class="rdma-host rdma-host-a">Server A</span>
    <span class="rdma-host rdma-host-b">Server B</span>
    <div class="rdma-track"><span class="rdma-record"></span></div>
    <span>Source<br>buffer</span>
    <span>Network<br>adapter</span>
    <span>Network</span>
    <span>Network<br>adapter</span>
    <span>Destination<br>buffer</span>
  </div>
  <ol class="rdma-steps" role="list">
    <li data-rdma-phase="prepare"><strong>Prepare a buffer.</strong> B registers a free buffer and gives A its address and remote key. B also puts a receive request in its queue to accept a notification.</li>
    <li data-rdma-phase="submit"><strong>Post the write.</strong> A tells its adapter where to read locally, how many bytes to copy, and where to place them on B.</li>
    <li data-rdma-phase="transfer"><strong>Move the bytes.</strong> A's adapter reads the buffer with DMA and sends the record. B's adapter checks the destination and writes the bytes into B's buffer.</li>
    <li data-rdma-phase="complete"><strong>Check the results.</strong> A can reuse its source buffer after a successful local completion. B gets a receive completion with the buffer's identifier, so its program knows where to read.</li>
    <li data-rdma-phase="persist"><strong>Save the record.</strong> B's program appends the record to its log and acknowledges it when the log is safely stored.</li>
  </ol>
  <div class="rdma-controls" hidden>
    <span class="rdma-status" role="status" aria-atomic="true"><span data-rdma-count></span><span class="rdma-announcement" data-rdma-announcement></span></span>
    <div class="rdma-actions">
      <button type="button" data-rdma-action="previous">Previous</button>
      <button type="button" data-rdma-action="play">Play</button>
      <button type="button" data-rdma-action="next">Next</button>
    </div>
  </div>
</figure>

The example uses **Write with Immediate**. A attaches a 32-bit number that identifies the destination buffer. B's adapter consumes the posted receive request to deliver that number in a completion. A plain Write leaves out that notification. The completion tells B that the bytes are ready in memory. B's later acknowledgment tells A that the application has saved the record. [Linux send and completion manuals][post] · [RDMA programming manual][programming]

## Read, write, and send

The question to ask is who picks the buffer on the other machine.

| Operation | What it does | Who picks the remote buffer? |
| --- | --- | --- |
| **RDMA Write** | Pushes bytes to another machine. | The writer names the destination. |
| **RDMA Read** | Pulls bytes into a local buffer. | The reader names the source. |
| **Send and Receive** | Delivers a message to a buffer the receiver has prepared. | The receiver supplies the destination. |

Read and Write are called **one-sided** because the remote program can authorize access ahead of time and leave the transfer to the adapters. Send and Receive are **two-sided** because the receiver supplies a receive request for each message. It can queue those requests in advance. [RDMA programming manual][programming]

## Which networks carry RDMA?

**InfiniBand** is a network built for workloads such as high-performance computing. **RDMA over Converged Ethernet (RoCE)** carries RDMA over Ethernet. Version 2, written **RoCEv2**, can cross Internet Protocol (IP) routers. Its packets are carried in User Datagram Protocol (UDP), and reliable RDMA connections have their own delivery and retry rules. The **Internet Wide Area RDMA Protocol (iWARP)** carries RDMA over TCP. Hardware and network configuration determine which transports are available. [InfiniBand programming manual][programming] · [RoCE guide][roce] · [Microsoft RDMA transport overview][iwarp] · [iWARP specification][mpa]

## Where you encounter RDMA

- **Windows file shares.** Microsoft's Server Message Block (SMB) Direct uses compatible adapters to move file data with less CPU work. This matters when a database or a virtual machine keeps its files on another server. [Microsoft SMB Direct documentation][smb]
- **Networked block storage.** Non-Volatile Memory Express (NVMe) over Fabrics lets a host access a remote drive as block storage. With its RDMA transport, read and write data move between the host's buffers and the storage server over the network. [NVMe over RDMA specification][nvme]
- **Distributed model training.** The NVIDIA Collective Communications Library (NCCL) helps graphics processing units (GPUs) combine results across machines. For example, workers can sum their gradients at each training step. NCCL can use InfiniBand or RoCE to exchange those arrays. [NVIDIA collective operations][collectives] · [NCCL network transport documentation][nccl]
- **Language model serving.** vLLM is an open-source engine for serving large language models. One worker can process the prompt while a separate worker generates the answer. The second worker can use the NVIDIA Inference Xfer Library (NIXL) to pull the key-value (KV) cache, the attention data calculated for the prompt, over RDMA. With supported hardware, GPUDirect RDMA lets the adapters access GPU memory directly, avoiding a copy through each machine's main memory. [vLLM cache transfer design][vllm] · [NIXL project][nixl] · [GPUDirect documentation][gpudirect]

RDMA is useful when moving data takes enough time or CPU capacity to matter, and the application can reuse registered buffers. It adds work too. Programs have to keep buffers valid while transfers are in flight, agree on when to reuse them, and handle failures. The network still has to carry every byte. For the log example, measure the time until B's durable acknowledgment and the CPU time used to get there. A fast transfer can free CPU capacity even when storage still accounts for most of the wait.

[dma]: https://docs.kernel.org/core-api/dma-api-howto.html
[tcp]: https://www.rfc-editor.org/rfc/rfc9293.html
[recv]: https://man7.org/linux/man-pages/man2/recv.2.html
[uverbs]: https://www.kernel.org/doc/html/latest/infiniband/user_verbs.html
[rdmap]: https://www.rfc-editor.org/rfc/rfc5040.html
[register]: https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html
[programming]: https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf#page=20
[poll]: https://man7.org/linux/man-pages/man3/ibv_poll_cq.3.html
[post]: https://man7.org/linux/man-pages/man3/ibv_post_send.3.html
[roce]: https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-512/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/
[iwarp]: https://learn.microsoft.com/en-us/windows-hardware/drivers/network/background-reading-on-rdma
[mpa]: https://www.rfc-editor.org/rfc/rfc5044.html
[smb]: https://learn.microsoft.com/en-us/windows-server/storage/file-server/smb-direct
[nvme]: https://nvmexpress.org/specification/rdma-transport-specification/
[collectives]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-disable
[vllm]: https://docs.vllm.ai/en/v0.31.0/design/nixl_kv_cache_lease/
[nixl]: https://github.com/ai-dynamo/nixl
[gpudirect]: https://docs.nvidia.com/cuda/gpudirect-rdma/
