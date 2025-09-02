/*
Aim of the program: Change the launch configuration to 8 blocks of 32 threads and verify the global IDs are still unique and correct.

Thinking Process:
1. The program will launch a simple CUDA kernel where each thread computes its global ID based on blockIdx.x and threadIdx.x.
2. We want to use 8 blocks with 32 threads per block, giving 256 total threads.
3. Each thread will write its global ID into an array at an index equal to that ID.
4. After kernel execution, the host will copy back the array and verify that for every index i, array[i] == i.
5. If all values match, the program will print that the global IDs are unique and correct. If any mismatch occurs, it will report the error.
6. The code will handle allocation, kernel launch, memory copy, and cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void compute_global_ids(int *global_ids, int total_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_threads) {
        global_ids[idx] = idx;
    }
}

int main(void) {
    const int threads_per_block = 32;
    const int num_blocks = 8;
    const int total_threads = threads_per_block * num_blocks;

    int *d_global_ids = NULL;
    int *h_global_ids = (int*)malloc(total_threads * sizeof(int));
    if (h_global_ids == NULL) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return 1;
    }

    cudaError_t err = cudaMalloc((void**)&d_global_ids, total_threads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_global_ids);
        return 1;
    }

    // Launch kernel
    compute_global_ids<<<num_blocks, threads_per_block>>>(d_global_ids, total_threads);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_global_ids);
        free(h_global_ids);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_global_ids, d_global_ids, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_global_ids);
        free(h_global_ids);
        return 1;
    }

    // Verify uniqueness and correctness
    int unique = 1;
    for (int i = 0; i < total_threads; ++i) {
        if (h_global_ids[i] != i) {
            printf("Mismatch at index %d: expected %d, got %d\n", i, i, h_global_ids[i]);
            unique = 0;
            break;
        }
    }

    if (unique) {
        printf("All global IDs are unique and correct.\n");
    } else {
        printf("Global ID verification failed.\n");
    }

    // Cleanup
    cudaFree(d_global_ids);
    free(h_global_ids);

    return 0;
}
