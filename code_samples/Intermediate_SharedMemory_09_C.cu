```cpp
/*
Aim of the program:
Write a kernel where `output[i] = shared_mem[threadIdx.x]`. This should be conflict‑free.

Thinking:
- To avoid shared‑memory bank conflicts we want each thread in a warp to access a
  different bank. CUDA shared memory has 32 banks; the bank index is
  (address / sizeof(T)) % 32.  If we use a contiguous array and each thread
  accesses shared_mem[threadIdx.x] where threadIdx.x goes from 0 to blockDim.x-1,
  then for a warp of 32 threads the accessed banks are 0,1,2,…,31 – one per
  thread – so there are no bank conflicts.
- The kernel will copy from global input to shared memory, then write back
  from shared memory to global output.  The assignment `output[i] = shared_mem[threadIdx.x]`
  is performed only if the global index i is within bounds.
- The host code sets up input data, allocates device memory, launches the kernel
  with `blockDim.x` threads and enough shared memory (`blockDim.x * sizeof(int)`).
- Simple error checking and a small output verification are included.

Overall, this example demonstrates a conflict‑free shared‑memory usage pattern
and fulfills the requirement that `output[i] = shared_mem[threadIdx.x]`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024            // Number of elements
#define BLOCK_SIZE 256    // Threads per block

// Kernel that copies input to shared memory and then to output.
// No shared‑memory bank conflicts because each thread accesses a unique address.
__global__ void copy_shared_conflict_free(const int* __restrict__ input,
                                          int* __restrict__ output,
                                          int n)
{
    // Allocate shared memory dynamically
    extern __shared__ int shared_mem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Write to shared memory
        shared_mem[threadIdx.x] = input[idx];
        // Read from shared memory and write to output
        output[idx] = shared_mem[threadIdx.x];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

int main(void)
{
    // Allocate host memory
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = i * 2;   // arbitrary data
    }

    // Allocate device memory
    int *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemBytes = BLOCK_SIZE * sizeof(int);
    copy_shared_conflict_free<<<gridSize, BLOCK_SIZE, sharedMemBytes>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 results:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("output[%d] = %d\n", i, h_output[i]);
    }

    // Cleanup
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
```