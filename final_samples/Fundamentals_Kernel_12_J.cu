In CUDA kernel launch configuration you usually compute the number of thread blocks as

```cpp
int numBlocks = (N + BS - 1) / BS;
```

rather than the simpler `N / BS`.  
The reason is that the division `N / BS` is an integer division that **truncates** the result.  
If `N` is not an exact multiple of the block size `BS`, the truncated quotient will be one block short, leaving the remaining `N % BS` elements unprocessed.

The expression `(N + BS - 1) / BS` works as follows:

1. `N + BS - 1` ensures that if there is any remainder when dividing `N` by `BS`, the addition pushes the sum up to the next multiple of `BS`.
2. The integer division then yields the ceiling of `N / BS`, i.e., the smallest integer `≥ N / BS`.

Thus, this formula guarantees that all `N` elements are covered by the kernel launch, even when `N` is not divisible by `BS`.