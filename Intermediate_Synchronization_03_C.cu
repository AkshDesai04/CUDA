/*
Run the program multiple times and show that the incorrect result is also non-deterministic.
Thought process:
1. The goal is to demonstrate non-deterministic behavior in a CUDA program by introducing a race condition.
2. A common way to do this is to have multiple threads update the same global memory location without synchronization primitives such as atomic operations.
3. The program below allocates a single integer on the device, initializes it to zero, and launches many threads that each increment this integer 1000 times.
4. Because the increment operation is not atomic, the final value of the counter will vary from run to run, revealing non-deterministic results.
5. The host prints the counter after the kernel completes. Running the program multiple times should show different values.
6. This code is written in C with CUDA extensions (.cu file). It uses CUDA runtime API functions: cudaMalloc, cudaMemcpy, cudaFree, and a simple kernel.
7. The kernel is intentionally incorrect (no atomic operation) to illustrate the issue.
8. The program is self-contained and can be compiled with `nvcc -o race_example race_example.cu` and then executed multiple times.
*/

#include <stdio.h>
#include <stdlib.h>

// Kernel that increments a global counter many times.
// The increment is not atomic, causing a race condition.
__global__ void race_kernel(int *counter, int increments_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < increments_per_thread; ++i) {
        // Non-atomic read-modify-write: race condition!
        int val = *counter;
        val = val + 1;
        *counter = val;
    }
}

int main(void) {
    const int num_threads = 1024;           // number of CUDA threads
    const int block_size = 256;             // threads per block
    const int increments_per_thread = 1000; // how many times each thread increments

    // Allocate device memory for counter
    int *d_counter;
    cudaError_t err = cudaMalloc((void **)&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize counter to zero on the device
    int h_counter = 0;
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (num_threads + block_size - 1) / block_size;
    race_kernel<<<num_blocks, block_size>>>(d_counter, increments_per_thread);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_counter);

    // Print result
    printf("Final counter value: %d\n", h_counter);
    return 0;
}
