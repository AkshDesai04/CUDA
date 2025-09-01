/*
Aim of the program:
Modify the program to launch the kernel with one block containing two threads.

Thinking process:
1. The requirement is to ensure that the kernel launch configuration uses exactly one block and two threads per block.
2. A minimal but functional CUDA program is created to demonstrate this launch configuration.
3. The kernel performs a simple operation (squaring a number) on an array of two elements.
4. The host code allocates memory on both host and device, copies data to the device, launches the kernel with <<<1, 2>>>, copies the results back, and prints them.
5. Error checking is added for robustness, but the focus remains on the kernel launch configuration.
6. The program is self-contained in a single .cu file, adhering to the instruction to output only the code file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that squares each element in the input array
__global__ void squareKernel(const float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(void) {
    const int N = 2;                      // Number of elements (matches two threads)
    size_t size = N * sizeof(float);

    // Host memory allocation
    float h_in[N]  = {1.0f, 2.0f};
    float h_out[N] = {0};

    // Device memory allocation
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel with 1 block and 2 threads
    dim3 blockSize(2);        // Two threads per block
    dim3 gridSize(1);         // One block
    squareKernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    // Check for any launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; ++i) printf("%f ", h_in[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; ++i) printf("%f ", h_out[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
