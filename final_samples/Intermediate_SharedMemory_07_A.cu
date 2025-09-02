```cpp
/*
Aim: Generalize this to reverse a large array by having each block reverse its own segment.

Thinking:
- Reversing a huge array efficiently on GPU requires parallel work distribution.
- The strategy is to partition the array into contiguous segments (chunks), each handled by a
  single CUDA block. Each block reverses its own chunk by swapping symmetric elements
  within that chunk.
- Let BLOCK_THREADS be the number of threads per block (e.g., 256).  
  Each block will work on a chunk of size CHUNK_SIZE = 2 * BLOCK_THREADS.  
  This guarantees that each thread can handle at most one swap pair per iteration,
  and any extra pairs are handled via a loop stride equal to blockDim.x.
- For the final block, the chunk may be smaller than CHUNK_SIZE; we clip the
  operations to the remaining number of elements, which is handled automatically
  by bounds checks in the kernel.
- Each thread executes a loop:
      for (size_t i = blockStart + threadIdx.x;
           i < blockStart + chunkSize/2 && i < N;
           i += blockDim.x)
  where `i` is the global index of the left element of a swap pair.
  The corresponding right index is `j = blockStart + chunkSize - 1 - (i - blockStart)`.
  We perform the swap only if `j` is within the array bounds.
- Host code:
  * Allocate and initialise a large array on the host.
  * Copy it to device memory.
  * Determine number of blocks: ceil(N / CHUNK_SIZE).
  * Launch kernel with that many blocks and BLOCK_THREADS threads per block.
  * Copy the reversed array back to the host and verify correctness.
- The kernel uses no shared memory; all operations are on global memory.
- This design keeps the memory footprint low and scales with array size.
- The code below implements the above plan with a small demonstration array
  (size 1,024 for readability).  In practice, the array size can be much larger
  (e.g., millions of elements) without modification.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel to reverse each segment of the array
__global__ void reverseSegments(int *d_arr, size_t N, size_t chunkSize)
{
    // Start index of this block's chunk
    size_t blockStart = blockIdx.x * chunkSize;

    // Each thread processes multiple swap pairs within its chunk
    for (size_t i = blockStart + threadIdx.x; 
         i < blockStart + chunkSize / 2 && i < N; 
         i += blockDim.x)
    {
        // Corresponding index on the right side of the chunk
        size_t j = blockStart + chunkSize - 1 - (i - blockStart);

        // Ensure j is within array bounds (for the last partial block)
        if (j >= N) continue;

        // Perform the swap
        int tmp = d_arr[i];
        d_arr[i] = d_arr[j];
        d_arr[j] = tmp;
    }
}

int main()
{
    const size_t N = 1024;                 // Array size (can be much larger)
    const int BLOCK_THREADS = 256;         // Threads per block
    const size_t CHUNK_SIZE = 2 * BLOCK_THREADS; // Elements per block

    size_t numBlocks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Allocate host memory
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize array with sample data
    for (size_t i = 0; i < N; ++i) {
        h_arr[i] = static_cast<int>(i);
    }

    // Allocate device memory
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    reverseSegments<<<numBlocks, BLOCK_THREADS>>>(d_arr, N, CHUNK_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify correctness (print first and last 10 elements)
    printf("Reversed array (first 10 elements):\n");
    for (size_t i = 0; i < 10 && i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    printf("Reversed array (last 10 elements):\n");
    for (size_t i = (N > 10 ? N - 10 : 0); i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
```