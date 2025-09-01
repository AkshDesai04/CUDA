/*
In the host code, use `cudaMemset` to initialize the input array `a` to a specific value (e.g., 1) on the device before launching the kernel.

Thinking:
- The user requests that we use cudaMemset in the host code to initialize the device array a to a specific value before kernel launch.
- cudaMemset sets each byte of the allocated device memory to the given value. Therefore, if we want each element of an integer array to be 1, we cannot simply call cudaMemset(d_a, 1, N * sizeof(int)), because that will set each byte to 0x01, resulting in the integer value 0x01010101 (16843009 decimal). To avoid confusion, we will allocate a char array where each element is a single byte. Then setting each byte to 1 is straightforward and will make each element equal to 1.
- After initializing with cudaMemset, we will launch a simple kernel that multiplies each element by 2. This demonstrates that the device array had the correct initial value.
- The host will then copy the result back and print a few values to confirm the operation.
- We include error checking for CUDA API calls and kernel launch.
- The program will compile with nvcc and run on any CUDA-capable device.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that multiplies each element by 2
__global__ void multiplyByTwo(unsigned char* d_a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_a[idx] *= 2;
    }
}

int main() {
    const int N = 16;                // Number of elements
    const int bytes = N * sizeof(unsigned char);

    // Allocate device memory
    unsigned char* d_a = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));

    // Initialize device array to 1 using cudaMemset
    // Each byte is set to 0x01, which makes each element equal to 1
    CUDA_CHECK(cudaMemset(d_a, 1, bytes));

    // Define grid and block dimensions
    const int threadsPerBlock = 8;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch errors

    // Copy result back to host
    unsigned char h_a[N];
    CUDA_CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result after multiplying by 2:\n");
    for (int i = 0; i < N; ++i) {
        printf("%u ", h_a[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));

    return 0;
}
