/*
What is the size limit for constant memory? (Typically 64KB).

This program demonstrates the usage of constant memory in CUDA and implicitly
shows that the typical limit is 64KB on many GPUs. The program defines a
__constant__ array of exactly 64,000 bytes (a bit less than 64KB to stay
well within the limit) and launches a kernel that reads values from this
constant memory. It then prints a few values to confirm that the data was
copied correctly. By compiling and running this code on a CUDA-capable
device, the user can verify that the constant memory region can indeed hold
at least 64KB of data. The choice of 64,000 bytes is intentional to avoid
overflow while staying close to the limit. This simple example also
illustrates how to declare constant memory, copy data to it from the host,
and access it within a kernel.

The program flow:
1. Allocate host array and fill with values.
2. Copy host array to device constant memory using cudaMemcpyToSymbol.
3. Launch a kernel that reads the constant memory array.
4. Kernel prints a few selected values.
5. Clean up and exit.

The aim of this code is to answer the question: "What is the size limit
for constant memory? (Typically 64KB)" by providing a concrete example
that uses constant memory up to that limit.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CONST_MEM_SIZE 64000  // 64,000 bytes (~63KB), comfortably within 64KB

// Declare a constant memory array of 64,000 bytes
__constant__ unsigned char d_constArray[CONST_MEM_SIZE];

// Kernel that reads from constant memory and prints some elements
__global__ void readConstKernel() {
    int idx = threadIdx.x;
    if (idx < 10) {  // just print first 10 elements for demonstration
        printf("Thread %d: constArray[%d] = %u\n", idx, idx, d_constArray[idx]);
    }
}

int main() {
    // Allocate and initialize host array
    unsigned char h_array[CONST_MEM_SIZE];
    for (int i = 0; i < CONST_MEM_SIZE; ++i) {
        h_array[i] = (unsigned char)(i % 256);
    }

    // Copy host array to device constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_constArray, h_array, CONST_MEM_SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel
    readConstKernel<<<1, 10>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Constant memory test completed.\n");
    return 0;
}
