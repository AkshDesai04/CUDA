```cpp
/*
Aim: Write a kernel where `output[i] = shared_mem[threadIdx.x * 32]`. This should cause severe 32-way bank conflicts. Profile it and compare to the stride-1 case.

Thinking:
- In CUDA, shared memory is divided into banks (32 banks on current GPUs). 
  Two accesses that target the same bank in the same warp cause a bank conflict.
- For integer addresses, bank = (address / sizeof(int)) % 32.
  If we use an index `threadIdx.x * 32`, then index % 32 == 0 for all threads,
  so every thread accesses the same bank -> 32‑way conflict.
- The stride‑1 kernel accesses `shared_mem[threadIdx.x]`; here indices 0…31 map to
  banks 0…31 respectively, giving a conflict‑free pattern.
- To observe the difference we create two kernels: one with the intentional
  conflict, one without.  Both write the value they read back into an output
  array for verification.
- Timing is done with CUDA events.  Each kernel is launched with dynamic
  shared memory of size `blockDim.x * 32 * sizeof(int)` to accommodate all
  accesses.
- We use a simple test vector of 1M elements (1 << 20) and a block size of
  32 threads.  This keeps the example small and easy to run on most GPUs.
- After execution we copy the output arrays back to host and verify that
  the data matches the expected values.
- The program prints the elapsed time (in milliseconds) for both kernels
  and the throughput in GB/s for comparison.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that causes 32-way bank conflicts
__global__ void kernel_bank_conflict(int *output)
{
    extern __shared__ int shared_mem[];

    // Compute global index
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread writes to its unique slot, but with stride 32
    int local_idx = threadIdx.x * 32;
    shared_mem[local_idx] = threadIdx.x;  // simple data

    __syncthreads(); // Ensure all writes are visible

    // Read back from the same slot
    output[gid] = shared_mem[local_idx];
}

// Kernel that accesses shared memory with stride 1 (no bank conflict)
__global__ void kernel_stride1(int *output)
{
    extern __shared__ int shared_mem[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int local_idx = threadIdx.x; // stride 1
    shared_mem[local_idx] = threadIdx.x;

    __syncthreads();

    output[gid] = shared_mem[local_idx];
}

int main()
{
    const int N = 1 << 20;                // 1M elements
    const int threadsPerBlock = 32;       // Block size to trigger conflicts
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedMemSize = threadsPerBlock * 32 * sizeof(int); // dynamic shared memory

    // Allocate device memory for outputs
    int *d_output_conflict = nullptr;
    int *d_output_stride1  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_conflict, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_stride1,  N * sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------- Run conflict kernel -----------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_bank_conflict<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_output_conflict);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_conflict = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_conflict, start, stop));

    // ----------------- Run stride-1 kernel -----------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_stride1<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_output_stride1);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_stride1 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_stride1, start, stop));

    // ----------------- Verify results -----------------
    int *h_output_conflict = (int*)malloc(N * sizeof(int));
    int *h_output_stride1  = (int*)malloc(N * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_output_conflict, d_output_conflict,
                          N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_stride1, d_output_stride1,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_output_conflict[i] != h_output_stride1[i]) {
            printf("Mismatch at %d: conflict=%d stride1=%d\n",
                   i, h_output_conflict[i], h_output_stride1[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("Result verification PASSED.\n");
    else printf("Result verification FAILED.\n");

    // ----------------- Print timings -----------------
    printf("Conflict kernel elapsed time: %f ms\n", ms_conflict);
    printf("Stride-1 kernel elapsed time : %f ms\n", ms_stride1);

    // Throughput calculation (GB/s)
    double gb_per_sec_conflict = (double)N * sizeof(int) / ms_conflict * 1e-3;
    double gb_per_sec_stride1  = (double)N * sizeof(int) / ms_stride1  * 1e-3;
    printf("Conflict kernel throughput: %f GB/s\n", gb_per_sec_conflict);
    printf("Stride-1 kernel throughput : %f GB/s\n", gb_per_sec_stride1);

    // ----------------- Clean up -----------------
    free(h_output_conflict);
    free(h_output_stride1);
    CUDA_CHECK(cudaFree(d_output_conflict));
    CUDA_CHECK(cudaFree(d_output_stride1));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```