```cuda
/*
    Aim of the program:
    Write a kernel where `output[i] = shared_mem[threadIdx.x]`. This should be conflict-free.

    Thinking:
    In CUDA, shared memory is divided into 32 banks. When threads within a warp access the same bank,
    a bank conflict occurs, which serializes the memory access.  If we directly use
    `shared_mem[threadIdx.x]` as the address, threads 0–31 each touch a different bank
    (0–31), but threads 32–63 will again map to banks 0–31, causing conflicts for
    warps beyond the first one.  To avoid this, we pad the shared memory array so that
    consecutive threads map to distinct banks in every warp.

    A simple way is to allocate twice as many entries and use an even index for each
    thread: `shared_mem[threadIdx.x * 2]`.  The bank index for a thread `i` becomes
    `(2*i) % 32`, which yields a unique bank for all 32 threads of any warp.
    This guarantees a conflict‑free access pattern for any number of threads per block.

    The kernel therefore:
        1. Loads `input[tid]` into `shared_mem[tid*2]`.
        2. Synchronizes threads to ensure all writes to shared memory are complete.
        3. Reads back from `shared_mem[tid*2]` and writes to `output[tid]`.

    Host code demonstrates the kernel with 256 threads per block and 4 blocks,
    using an array size of 1024.  The shared memory size per block is
    `blockDim.x * 2 * sizeof(int)` to accommodate the padding.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024            // Total number of elements
#define THREADS_PER_BLOCK 256
#define BLOCKS (N / THREADS_PER_BLOCK)

// Kernel that copies input to output via padded shared memory (conflict‑free)
__global__ void copyViaShared(const int *input, int *output)
{
    extern __shared__ int shared_mem[];          // Shared memory declaration
    int tid = threadIdx.x;                      // Thread index within the block

    // Use padded index to avoid bank conflicts: each thread writes to an even address
    int smem_idx = tid * 2;

    // Load from global memory to shared memory
    if (blockIdx.x * blockDim.x + tid < N) {
        shared_mem[smem_idx] = input[blockIdx.x * blockDim.x + tid];
    } else {
        // Handle out-of-bounds (not expected with current parameters)
        shared_mem[smem_idx] = 0;
    }

    __syncthreads();  // Ensure all writes to shared memory are complete

    // Read back from shared memory to global memory
    if (blockIdx.x * blockDim.x + tid < N) {
        output[blockIdx.x * blockDim.x + tid] = shared_mem[smem_idx];
    }
}

int main()
{
    int h_input[N];
    int h_output[N];

    // Initialize input with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
    }

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_output, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate shared memory size per block (padded)
    size_t shared_mem_size = THREADS_PER_BLOCK * 2 * sizeof(int);

    // Launch kernel
    copyViaShared<<<BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i]) {
            printf("Mismatch at %d: got %d, expected %d\n", i, h_output[i], h_input[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("All values match! Kernel executed conflict‑free.\n");
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```