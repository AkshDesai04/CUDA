/*
 * Aim of the program: Implement the AoS version: create MyStruct* d_structs. Write a kernel that reads only the x component from each struct: output[i] = d_structs[i].x.
 *
 * Thinking:
 *  - We define a simple struct `MyStruct` that contains three float members: x, y, and z.
 *  - The host will allocate and fill an array of `MyStruct` objects with dummy data.
 *  - We allocate corresponding memory on the device for the struct array (`d_structs`) and for an output array of floats (`d_output`).
 *  - The CUDA kernel `extractX` will compute its global index, read the `x` component of the struct at that index, and write it to the output array.
 *  - After kernel execution we copy the output back to the host and print it to verify correctness.
 *  - All CUDA API calls are wrapped with a simple error-checking macro to catch and report failures early.
 *  - The code uses the classic C-style CUDA API and is written in a single .cu file, ready to be compiled with `nvcc`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

typedef struct {
    float x;
    float y;
    float z;
} MyStruct;

// Kernel that extracts the 'x' component from an array of structs
__global__ void extractX(const MyStruct* d_structs, float* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_structs[idx].x;
    }
}

int main(void) {
    const int N = 1024;          // Number of elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and initialize host array of structs
    MyStruct* h_structs = (MyStruct*)malloc(N * sizeof(MyStruct));
    if (!h_structs) {
        fprintf(stderr, "Failed to allocate host memory for structs.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_structs[i].x = (float)i;
        h_structs[i].y = (float)(i * 2);
        h_structs[i].z = (float)(i * 3);
    }

    // Allocate host output array
    float* h_output = (float*)malloc(N * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for output.\n");
        free(h_structs);
        return EXIT_FAILURE;
    }

    // Allocate device memory for structs and output
    MyStruct* d_structs = NULL;
    float* d_output = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_structs, N * sizeof(MyStruct)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy structs from host to device
    CHECK_CUDA(cudaMemcpy(d_structs, h_structs, N * sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Launch kernel
    extractX<<<blocksPerGrid, threadsPerBlock>>>(d_structs, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify and print a few results
    printf("First 10 extracted x components:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_structs[%d].x = %f -> output[%d] = %f\n", i, h_structs[i].x, i, h_output[i]);
    }

    // Cleanup
    free(h_structs);
    free(h_output);
    CHECK_CUDA(cudaFree(d_structs));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
