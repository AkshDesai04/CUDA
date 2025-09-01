/*
Pass the scalar via constant memory.

In this program we demonstrate how to use CUDA constant memory to pass a scalar value from the host
to a device kernel. Constant memory is read‑only on the device and is cached, so it is efficient for
broadcasting a single value (like a coefficient or a flag) to many threads. We declare a __constant__
variable on the device, copy the scalar to it from the host using cudaMemcpyToSymbol, and then
use it inside a simple kernel that scales each element of an input array. The host code allocates
memory, initializes data, launches the kernel, and retrieves the results. This illustrates the
basic pattern for using constant memory for scalars in CUDA programming.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device constant memory declaration for a scalar
__constant__ float d_scalar;

// Kernel that multiplies each element of the input array by the scalar in constant memory
__global__ void scaleKernel(const float *d_in, float *d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = d_in[idx] * d_scalar;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // Device allocations
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Host scalar value to be passed via constant memory
    float h_scalar = 2.5f;

    // Copy scalar to constant memory
    cudaMemcpyToSymbol(d_scalar, &h_scalar, sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_in[%d] = %f, h_out[%d] = %f\n", i, h_in[i], i, h_out[i]);
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
