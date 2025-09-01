/*
Aim: Pass an array of these structs from host to device. Have each thread i access the i-th element of the struct array and print a member.

Thinking:
- We need a struct definition that contains at least one member that we can print. We'll use:
    struct MyStruct { int a; float b; };
  The 'a' field will be printed by each thread.
- Host side:
  1. Allocate an array of MyStruct on the host and populate it.
  2. Allocate device memory with cudaMalloc.
  3. Copy the host array to device memory with cudaMemcpy.
- Device side:
  1. Define a kernel that takes a pointer to MyStruct and the array length.
  2. Each thread calculates its global index `i = blockIdx.x * blockDim.x + threadIdx.x`.
  3. If `i` is within bounds, use `printf` to output the member (e.g., `a`).
- After launching the kernel, we synchronize and free device memory.
- We'll add basic error checking after CUDA API calls.
- The program is self‑contained and compiles with nvcc as a .cu file.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple struct with two members.
struct MyStruct {
    int a;
    float b;
};

// Kernel: each thread prints the 'a' member of the i-th element.
__global__ void printStructKernel(const MyStruct *d_array, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Use printf to output from the device.
        printf("Thread %d: a = %d, b = %f\n", i, d_array[i].a, d_array[i].b);
    }
}

// Utility macro for error checking.
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(void)
{
    const int N = 10; // Number of elements
    const int threadsPerBlock = 4;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host array
    MyStruct *h_array = (MyStruct*)malloc(N * sizeof(MyStruct));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_array[i].a = i;
        h_array[i].b = i * 1.1f;
    }

    // Allocate device memory
    MyStruct *d_array = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_array, N * sizeof(MyStruct)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_array, h_array, N * sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Launch kernel
    printStructKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for device to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clean up
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);

    return 0;
}
