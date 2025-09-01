/*
Create an error by forgetting the `<<<...>>>` syntax for the kernel launch.

I am following the instruction to deliberately create a compile-time error in CUDA code by calling a kernel without the necessary launch configuration syntax (`<<<...>>>`). The program will define a simple CUDA kernel that increments each element of an array. The main function will allocate memory on the device, initialize data, and then attempt to launch the kernel. Instead of using the correct syntax `increment<<<gridDim, blockDim>>>(d_data);`, I will call it as a normal function `increment(d_data);`. This omission will cause the compiler to flag an error because CUDA kernels cannot be invoked like ordinary CPU functions. The code includes minimal error checking for memory allocation but intentionally omits the launch syntax to meet the requirement. The resulting error will be a clear compiler message indicating that a kernel launch configuration was expected. This demonstrates a common mistake that developers might make when transitioning between CPU and GPU programming.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernel that increments each element of the array by 1
__global__ void increment(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    const int N = 256;
    const int SIZE = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int*)malloc(SIZE);
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Incorrect kernel launch: missing <<<...>>> syntax
    increment(d_data, N);  // This line will cause a compile-time error

    // Copy data back to host (this part will not be reached due to the error)
    err = cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print result
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_data);
    free(h_data);
    return 0;
}
