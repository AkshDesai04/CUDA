A **shared memory bank** is a logical division of the CUDA device’s shared memory that allows multiple threads to read from or write to shared memory concurrently.  
- **Banking**: The shared memory is partitioned into *banks* (typically 32 banks on current NVIDIA GPUs). Each bank can serve one access per clock cycle without conflict.  
- **Bank conflicts**: If two or more threads in a warp access different addresses that map to the same bank, those accesses are serialized, reducing performance.  
- **Bank layout**: The address of a shared memory location is mapped to a bank by taking the address modulo the number of banks. Thus, proper data layout and access patterns are crucial to avoid conflicts.

In short, a shared memory bank is a sub‑section of shared memory that can be accessed in parallel by threads; GPUs usually provide **32 banks**.