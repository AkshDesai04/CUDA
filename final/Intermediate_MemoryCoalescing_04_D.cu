Yes – shared memory can be used to reorder the memory accesses so that the device performs only coalesced loads and stores.  
A typical pattern is:

1. **Launch a block** that owns a contiguous segment of the output array.  
2. **Thread‑to‑thread mapping**  
   * Each thread in the block is responsible for one (or a few) output elements.  
   * The thread indices are contiguous, so a store from shared memory to the output will be fully coalesced.  
3. **Strided read into shared memory**  
   * The data that needs to be written is not laid out contiguously in memory (e.g., it is interleaved or stored in a different order).  
   * Each thread performs a *strided* read: it reads the element that belongs to its output index from the global memory address that is offset by a stride (often the size of the problem or the number of blocks).  
   * These reads are **non‑coalesced** because neighboring threads access memory that is far apart.  
4. **Store into shared memory**  
   * The strided values are written into a shared‑memory buffer at an index that matches the thread index (e.g., `sdata[tid] = value;`).  
   * Because all threads write to unique shared‑memory locations that are contiguous, the write is efficient.  
5. **Synchronize**  
   * `__syncthreads();` ensures all threads have finished writing to shared memory before any thread reads from it.  
6. **Coalesced write to global memory**  
   * Each thread now reads its value from `sdata[tid]` and writes it to the global output array at the correct position (`output[global_index] = sdata[tid];`).  
   * Since `global_index` is contiguous for the block, the store is fully coalesced.

**Key points of the strategy**

| Step | Purpose | Memory access type |
|------|---------|--------------------|
| Strided read | Gather needed elements | Non‑coalesced |
| Write to shared | Temporary buffer | Fast, local |
| `__syncthreads()` | Ensure data availability | N/A |
| Coalesced write | Store final results | Coalesced |

By using shared memory in this way, you absorb the non‑coalesced strided accesses into a fast on‑chip buffer, allowing the expensive global memory writes to be performed in a single, coalesced transaction. This often results in a substantial speed‑up, especially when the striding pattern causes many misaligned or scattered accesses.