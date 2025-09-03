/*
Aim: Explain what a 'memory fence' is and how it relates to debugging memory errors.

Thinking:
- A memory fence (also called a memory barrier) is a synchronization primitive that enforces ordering constraints on memory operations. In CUDA, there are several types of fences:
  * __syncthreads(): a barrier that synchronizes all threads within a block and ensures that all shared memory writes are visible to all threads after the barrier.
  * __threadfence(): a device-wide memory fence that guarantees that all global memory writes issued by a thread before the fence become visible to all other threads on the device after the fence.
  * __threadfence_block(): similar to __threadfence() but only guarantees visibility within the block.
  * __threadfence_system(): guarantees that writes are visible to the host as well as to other devices.

- These fences are crucial for debugging memory errors because:
  * They prevent race conditions by ensuring that writes are completed before reads occur.
  * Without proper fences, a thread might read stale or partially updated data, leading to subtle bugs that are hard to reproduce.
  * In debugging, inserting fences can help identify where incorrect ordering or missing synchronization occurs.

- In this CUDA program, we illustrate the use of __threadfence() by having a kernel that writes to a global array and then reads it back from the host. The fence ensures that the write is committed before the host reads the data, preventing potential memory consistency issues.

- The code is intentionally simple: a kernel writes an incrementing value into a global array, fences, and the host verifies the result. The host also prints the data to demonstrate that the fence worked correctly.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that writes values to global memory and uses a device-wide fence
__global__ void write_and_fence(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Perform some computation (here just assign idx)
        d_arr[idx] = idx;

        // Ensure that this write is visible to all threads and host before proceeding
        __threadfence();  // Device-wide memory fence
    }
}

int main() {
    const int N = 1024;
    int *d_arr;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    write_and_fence<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy result back to host
    int *h_arr = (int*)malloc(size);
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Verify the result
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i) {
            printf("Mismatch at index %d: expected %d, got %d\n", i, i, h_arr[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("Memory fence test succeeded: all values are correct.\n");
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
