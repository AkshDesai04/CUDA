/*
Pass a size of 0 to `cudaMemcpy`. Is this an error?

My reasoning:
- According to the CUDA Runtime API, cudaMemcpy takes a size argument in bytes.
- If size is 0, the function should effectively do nothing (no data transfer).
- The API documentation states that passing a size of 0 is allowed and should return cudaSuccess.
- We will test this by calling cudaMemcpy with a 0 size and checking the returned error code.
- We'll allocate some dummy memory, but not copy anything.
- We expect the function to return cudaSuccess, confirming that size 0 is not an error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Allocate some host memory
    size_t hostSize = 10;
    int *h_data = (int*)malloc(hostSize * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    int *d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, hostSize * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_data);
        return -1;
    }

    // Perform cudaMemcpy with size 0 (Host to Device)
    err = cudaMemcpy(d_data, h_data, 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HtoD with size 0 failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaMemcpy HtoD with size 0 succeeded (expected).\n");
    }

    // Perform cudaMemcpy with size 0 (Device to Host)
    err = cudaMemcpy(h_data, d_data, 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoH with size 0 failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaMemcpy DtoH with size 0 succeeded (expected).\n");
    }

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return 0;
}
