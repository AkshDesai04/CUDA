/*
Test your program with N = 1 and a block size of 256. Does it work correctly?
[Thinking]
The aim is to verify that a CUDA kernel works correctly when the array size N is 1 and the block size is 256. 
A simple kernel that writes the thread index into the output array is sufficient for this test. 
The host program will launch the kernel with the appropriate number of blocks (ceil(N/blockSize)), 
allocate memory on both host and device, copy data back, and print the result. 
Because N=1, only the first thread (index 0) should write to d_out[0]. 
After copying back, the host should see d_out[0] == 0. 
If this is correct, the program works as expected for this edge case. 
All necessary CUDA error checks are included, and the program is fully self‑contained.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                  \
                      << "' in line " << __LINE__ << ": "                    \
                      << cudaGetErrorString(err) << std::endl;               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel that writes the global thread index into the output array
__global__ void testKernel(int *d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = idx;
    }
}

int main() {
    const int N = 1;               // Number of elements
    const int blockSize = 256;     // Threads per block

    // Calculate number of blocks needed
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Host memory allocation
    int *h_out = new int[N];

    // Device memory allocation
    int *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Launch kernel
    testKernel<<<numBlocks, blockSize>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result for N = " << N << ": ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_out;

    return 0;
}
