/*
Aim: Modify the program to launch with 128 threads instead of 64.

Thinking:
- The original prompt asked for a simple CUDA program that runs a kernel with 64 threads.
- To satisfy the requirement, we change the launch configuration to use 128 threads per block.
- The program will still compile and run correctly, demonstrating that the kernel launches with the new thread count.
- For clarity, we keep the kernel simple: each thread writes its index to an output array.
- The host allocates memory, copies data back, prints a few elements to verify.
- No external dependencies beyond standard CUDA headers.
- The code is fully self-contained in a single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeThreadIndex(int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx;
    }
}

int main() {
    const int N = 256; // number of elements to process
    int *h_output = (int*)malloc(N * sizeof(int));
    int *d_output;
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Launch kernel with 128 threads per block
    const int threadsPerBlock = 128;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    writeThreadIndex<<<blocksPerGrid, threadsPerBlock>>>(d_output, N);

    cudaDeviceSynchronize(); // ensure kernel finished

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first 10 results to verify
    printf("First 10 thread indices written by kernel:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    cudaFree(d_output);
    free(h_output);

    return 0;
}
