```cuda
/* 
Launch the kernel with 10 threads, but have all 10 threads try to write to `data[0]`. 
What value do you expect to see when you copy it back? (This is a race condition). 

Thinking: 
- The kernel is launched with 10 threads in a single block. 
- All threads write to the same memory location `data[0]`. 
- Because the writes are not synchronized and no atomic operation is used, the writes 
  will race with each other. 
- The final value of `data[0]` after the kernel completes will be whatever value was 
  written by the last thread that the hardware happens to execute last. 
- Since thread scheduling on the GPU is non-deterministic for these operations, 
  the result is unpredictable and could be any of the values written by the threads 
  (e.g., 0 through 9). This demonstrates a classic race condition.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel where each thread writes its thread index to data[0]
__global__ void writeToData(int *data) {
    // All threads write to the same memory location
    data[0] = threadIdx.x;
}

int main(void) {
    const int ARRAY_SIZE = 1;
    const int NUM_THREADS = 10;

    // Host array
    int h_data[ARRAY_SIZE] = {0};

    // Device pointer
    int *d_data = NULL;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_data, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy host to device (not really needed here but kept for completeness)
    err = cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel with 10 threads
    writeToData<<<1, NUM_THREADS>>>(d_data);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the result
    printf("Value of data[0] after kernel execution: %d\n", h_data[0]);

    // Free device memory
    cudaFree(d_data);

    return 0;
}
```