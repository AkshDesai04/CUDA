When a warp (32 threads) performs a *strided read* each thread accesses an element that is `STRIDE` elements away from the previous thread’s element.  
Assuming each element is 4 bytes (32‑bit word) and that the GPU uses 128‑byte memory transactions, the number of transactions is determined by how the accessed addresses span these 128‑byte “banks”.

### STRIDE = 2
- Thread 0 reads element 0  → address 0  
- Thread 1 reads element 2  → address 8  
- Thread 2 reads element 4  → address 16  
- …  
- Thread 31 reads element 62 → address 248  

The addresses covered by the warp are from 0 to 248 bytes, i.e. a continuous span of 256 bytes.  
A 128‑byte transaction covers 0–127 and another covers 128–255.  
Therefore the warp generates **2 memory transactions** (one for the first 128‑byte block, one for the second).

### STRIDE = 32
- Thread 0 reads element 0   → address 0  
- Thread 1 reads element 32  → address 128  
- Thread 2 reads element 64  → address 256  
- …  
- Thread 31 reads element 992 → address 992  

Each thread accesses a separate 128‑byte block: thread 0 → 0–127, thread 1 → 128–255, …, thread 31 → 960–1023.  
Because every thread’s word is in a distinct 128‑byte segment, the warp must issue a **separate 128‑byte transaction for each thread**, i.e. **32 transactions**.

So:

| STRIDE | Number of 128‑byte memory transactions issued by the warp |
|--------|-----------------------------------------------------------|
| 2      | 2                                                         |
| 32     | 32                                                        |