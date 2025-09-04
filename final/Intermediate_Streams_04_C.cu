Using `cudaDeviceSynchronize()` after every kernel launch forces the host CPU to wait until **all** pending work on the GPU is finished before it can continue.  
This has a couple of disadvantages:

| What happens with `cudaDeviceSynchronize()` | What happens when you synchronize only the streams that need it |
|-------------------------------------------|--------------------------------------------------------------|
| The host blocks until *every* stream completes. | The host continues executing while kernels in the unrelated streams run concurrently. |
| The GPU is idle only for the host‑side idle time. | The GPU remains busy executing all streams simultaneously. |
| A dependency on the *global* device state must be re‑established each time. | Each stream has its own dependency chain, so the GPU only has to enforce the minimal ordering needed. |
| If a kernel in stream A blocks on the result of a kernel in stream B, the CPU will wait for both, even if the CPU does not need the data from stream B. | The CPU can wait only on stream A (or the specific stream that needs the result), keeping the other stream free. |

Thus, synchronizing only the necessary streams is more efficient because:

1. **Host CPU is not stalled** – the CPU can issue other work or run other parts of the program while the GPU continues executing kernels in streams that are independent of the current dependency.
2. **GPU stays busy** – all streams run concurrently, so the device does not sit idle waiting for the host to call `cudaDeviceSynchronize()`.
3. **Fine‑grained ordering** – the GPU enforces synchronization only where data hazards exist, allowing more overlapping of computation and communication.

In short, limiting synchronization to the streams that actually depend on each other keeps both the CPU and GPU active and avoids unnecessary stalls, leading to better overall performance.