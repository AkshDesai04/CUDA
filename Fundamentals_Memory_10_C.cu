In CUDA, a 2‑D allocation created with **`cudaMallocPitch`** returns a *pitch* (also called *pitch‑width* or *row stride*) that may be **larger than `width * sizeof(element)`**.  
The main reason for this extra padding is **memory alignment** to satisfy the hardware’s coalescing rules and to keep each row of the array aligned on a boundary that is optimal for device memory transactions.

### Alignment for Coalescing

1. **Coalesced Memory Accesses**  
   CUDA hardware fetches data in large, fixed‑size units (for example, 128‑byte or 256‑byte aligned cache lines).  
   When threads in a warp read a contiguous segment of memory, the accesses are merged into one or a few of these wide transactions only if the starting address is aligned to the transaction size and the width of the segment does not cross the alignment boundary.

2. **Row‑by‑Row Allocation**  
   For a 2‑D array, each row is stored sequentially in device memory.  
   If the row stride (`pitch`) were just `width * sizeof(element)`, a warp that reads an element from the first column of several consecutive rows might cross the boundary between rows.  
   That would cause the hardware to issue separate transactions for each row, breaking coalescing.

3. **Padding the Pitch**  
   `cudaMallocPitch` therefore rounds the requested width up to the nearest multiple of a **platform‑dependent alignment factor** (commonly 32, 64, 128, or 256 bytes).  
   The returned pitch is the smallest such multiple that can accommodate all elements in a row.  
   Thus, `pitch ≥ width * sizeof(element)` and the difference is padding added to keep each row start aligned.

4. **Benefit**  
   Although the padding increases the total memory footprint, it **improves memory throughput** by enabling the hardware to service wide, aligned memory requests efficiently.  
   For high‑performance kernels, this trade‑off is essential.

### Example

Suppose you allocate a 2‑D array of `float` (4 bytes each) with width = 100.  
`width * sizeof(float)` = 400 bytes.  
If the alignment requirement is 128 bytes, the next multiple of 128 that is ≥ 400 is 512.  
So `cudaMallocPitch` will return a pitch of **512** bytes.  
Each row will thus start 512 bytes apart, ensuring that a warp reading a contiguous block of `float`s in a row can be served by a single 128‑byte transaction.

In summary, **the pitch is padded to satisfy the hardware’s alignment constraints for coalesced memory accesses, which may make it larger than the raw data size of a row.**