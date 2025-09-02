If you launch a kernel that reads from a constant‑memory symbol that you never initialized (e.g., you forgot to call `cudaMemcpyToSymbol` or a similar copy routine), the device will simply read whatever bytes happen to be resident in that portion of constant memory at the time the kernel executes.  

There is **no automatic “set to zero”** semantic for constant memory the way you might expect from C++ static objects or CUDA’s `__shared__` memory that is automatically zero‑initialized each block. The contents of constant memory are undefined until you explicitly write to them. In practice, this means:

| Scenario | Typical outcome |
|----------|-----------------|
| **No previous write** | The kernel will see “garbage” values – whatever was left in that memory region by the driver, previous kernels, or other allocations. |
| **Previous writes from another kernel** | It will see those stale values (which might still be valid or might be outdated). |
| **Driver reset / fresh context** | Some drivers zero‑initialize device memory on reset, but you should **not** rely on this; the CUDA programming guide explicitly states that global and constant memory are *uninitialized* by default. |

Because the value is undefined, any computation that depends on that constant symbol can produce incorrect results or even cause the kernel to misbehave (e.g., division by zero, out‑of‑range indices, etc.). Therefore, always initialize constant memory before launching kernels that read from it.