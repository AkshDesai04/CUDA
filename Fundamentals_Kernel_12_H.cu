```cpp
/*
A kernel can optionally take shared memory size and a stream as launch parameters: `kernel<<<grid, block, shared_mem_bytes, stream>>>();`. Practice by setting the shared memory size to 0 and the stream to `0` (default stream).
This program demonstrates the use of optional launch parameters in CUDA. It defines a simple kernel that writes the index of each thread to an output array. In the launch configuration, the shared memory size is explicitly set to 0 (indicating no dynamic shared memory is needed) and the stream parameter is set to 0, which corresponds to the default stream. The program allocates device memory, launches the kernel with these parameters, copies the results back to the host, and prints them to verify correct execution. This example showcases how to use the third and fourth template arguments in the kernel launch syntax while keeping the code straightforward for educational purposes.
*/

#include <cuda_runtime.h>
#include <iostream>

__global__ void writeIndices(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

int main() {
    const int N = 10;
    int *d_out;
    int *h_out = new int[N];

    // Allocate device memory
    cudaMalloc(&d_out, N * sizeof(int));

    // Define grid and block dimensions
    dim3 block(4);
    dim3 grid((N + block.x - 1) / block.x);

    // Launch kernel with shared memory size = 0 and stream = 0
    writeIndices<<<grid, block, 0, 0>>>(d_out, N);

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_out);
        delete[] h_out;
        return EXIT_FAILURE;
    }

    // Copy results back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Kernel output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}
```