/*
Use Nsight Systems (`nsys`) to visualize the execution timeline. You should see a serial pattern: [COPY]...[KERNEL]...[COPY]...

I decided to create a minimal CUDA program that performs a host-to-device copy, launches a kernel that simply increments each element of an array, and then copies the result back to the host. This sequence of operations—two memory copies surrounding a kernel launch—will generate a clear serial pattern in the Nsight Systems timeline: first the data transfer to the device, then the kernel execution, and finally the transfer back to the host. I included basic error checking for each CUDA API call to ensure correctness and added a simple printout of a few elements to verify that the kernel ran. The code uses standard CUDA runtime API calls and is self-contained in a single .cu file, ready to be compiled with nvcc and profiled with nsys.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error in " << __FILE__            \
                      << " at line " << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)

__global__
void incrementKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // Host to device copy
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Device to host copy
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
    std::cout << "Sample output:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;

    return 0;
}
