In the naive implementation each thread reads an element of **B** using an index that jumps by the matrix width.  
For a row‑major stored matrix \(B\) of size \(N \times N\) the address of element \((i,j)\) is

\[
\text{addr}(i,j) = \text{base} + (i \times N + j) \times \text{sizeof(T)} .
\]

If the kernel launches a thread per element and the threads are numbered consecutively along the *row* dimension, then consecutive threads in a warp will read addresses

\[
\text{addr}(i,0), \text{addr}(i,1), \text{addr}(i,2), \dots
\]

which are contiguous.  
However, in the typical naive matrix‑multiplication kernel the outer loop is over rows of **A** (or columns of **B**) and the inner loop iterates over columns of **A**/rows of **B**.  
Thus a thread that works on a fixed *row of A* will read elements of **B** along a *column*:

```
for (int k = 0; k < N; ++k)
    sum += A[row][k] * B[k][col];
```

Here, when `col` is constant for a warp and `k` varies, consecutive threads read `B[0][col]`, `B[1][col]`, `B[2][col]`, … – i.e., memory addresses that differ by `N*sizeof(T)`.  
This produces a **strided access pattern** with stride `N`. The warp’s memory transaction is split into many small, unaligned transactions instead of a single wide transaction. As a result:

1. **Uncoalesced reads** – the GPU must issue many separate memory loads for a warp, instead of one efficient coalesced load.
2. **Reduced bandwidth** – the effective memory throughput drops dramatically because each transaction carries little useful data.
3. **Higher latency** – the memory controller must handle more requests, increasing stall time.

Therefore, the naive kernel’s access to matrix **B** is inefficient due to the strided, uncoalesced pattern that prevents the GPU from exploiting its memory bandwidth fully.