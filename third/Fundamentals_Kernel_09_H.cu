/*
Aim of the program: Create a `__device__` function that takes a pointer as an argument.

Thinking:
- The requirement is to demonstrate a __device__ function that accepts a pointer argument. 
- To make the example complete and compilable, we also provide a simple kernel that calls this device function, and host code that sets up data, launches the kernel, and retrieves the result. 
- The device function `incrementValue` will take an `int*` pointer and increment the integer it points to. 
- The kernel `launchIncrement` will receive an array pointer and call `incrementValue` on each element. 
- The host program will allocate an array of integers, copy it to device memory, launch the kernel with one thread per element, copy the results back, and print them. 
- This shows how pointers can be passed to device functions and how memory operations are performed in CUDA C. 
- We use `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and error checking macros to keep the code robust. 
- All code is contained in a single .cu file as requested. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ void incrementValue(int *ptr) {
    // Increment the integer that the pointer points to
    *ptr += 1;
}

__global__ void launchIncrement(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Call the device function with a pointer to the current element
        incrementValue(&data[idx]);
    }
}

int main() {
    const int N = 10;
    int h_data[N];
    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    int *d_data;
    size_t size = N * sizeof(int);
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HtoD failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    launchIncrement<<<blocks, threadsPerBlock>>>(d_data, N);

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print results
    printf("Results after incrementing each element:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_data);
    return 0;
}
