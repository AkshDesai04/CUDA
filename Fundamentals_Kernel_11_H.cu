/*
In a kernel with many threads, have every thread try to modify the same member of the single struct instance (e.g., `d_data->a = threadIdx.x;`). What is the final value after copying back? (Race condition).

The goal of this program is to illustrate a classic race condition in CUDA.  
We allocate a single structure on the device, launch a large number of
threads, and have every thread write its `threadIdx.x` value into the same
field (`a`) of that structure. Since there is no synchronization or atomic
operation, the order in which the writes reach the memory is undefined.
After the kernel finishes, the host copies the structure back and prints
the final value of `a`. That value can be any of the thread indices that
executed last, and different program runs (or different hardware) may
produce different results. This demonstrates that without proper
synchronization, concurrent writes to the same memory location lead to
nondeterministic behavior.

The program also contains minimal error checking and demonstrates the
basic CUDA memory transfer workflow: allocation, kernel launch, copy back,
and cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Data {
    int a;
};

__global__ void raceKernel(Data *d_data) {
    // All threads write their threadIdx.x to the same member 'a'
    d_data->a = threadIdx.x;
}

int main() {
    Data h_data;
    Data *d_data;
    cudaError_t err;

    // Allocate device memory for a single Data instance
    err = cudaMalloc((void**)&d_data, sizeof(Data));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with many threads (e.g., 256 threads in one block)
    const int threadsPerBlock = 256;
    raceKernel<<<1, threadsPerBlock>>>(d_data);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_data, d_data, sizeof(Data), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the final value of 'a'
    printf("Final value of d_data->a after race: %d\n", h_data.a);

    // Clean up
    cudaFree(d_data);
    return 0;
}
