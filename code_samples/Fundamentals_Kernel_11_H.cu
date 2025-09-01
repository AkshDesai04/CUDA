```cpp
/*
Aim of the program: In a kernel with many threads, have every thread try to modify the same member of the single struct instance (e.g., `d_data->a = threadIdx.x;`). What is the final value after copying back? (Race condition).

Thinking:
1. Define a simple struct containing one integer field.
2. Allocate one instance of this struct on the host and on the device.
3. Launch a CUDA kernel with many threads (e.g., 256).
4. Inside the kernel, each thread writes its own threadIdx.x into the same struct member `a`.
5. Because the writes are not synchronized, this is a classic race condition.
6. After kernel execution, copy the struct back to the host and print the value of `a`.
7. The printed value will be nondeterministic and typically will equal the value written by one of the last threads that finished execution.
8. The program demonstrates that without proper synchronization, concurrent writes to the same memory location can lead to unpredictable results.
*/

#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {
    int a;
} MyData;

// Kernel where every thread writes its threadIdx.x to the same struct member
__global__ void raceKernel(MyData *d_data) {
    // Each thread attempts to modify the same field
    d_data->a = threadIdx.x;
}

int main() {
    MyData h_data;
    MyData *d_data = NULL;
    size_t size = sizeof(MyData);

    // Initialize host data
    h_data.a = 0;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy host data to device
    err = cudaMemcpy(d_data, &h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel with 256 threads (one block)
    int threadsPerBlock = 256;
    raceKernel<<<1, threadsPerBlock>>>(d_data);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(&h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print the final value of 'a' after the race
    printf("Final value of d_data->a after kernel: %d\n", h_data.a);

    // Clean up
    cudaFree(d_data);
    return 0;
}
```