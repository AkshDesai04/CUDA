/*
Aim: Use a struct as the type for your __constant__ variable. Copy a host struct to it and access its members in the kernel.

Thinking:
- Define a simple struct `Params` with a couple of fields (an integer offset and a floating-point scale).
- Declare a `__constant__` variable of type `Params` named `d_params`.
- On the host, create an instance of `Params`, fill it with some values, and copy it to the device constant memory using `cudaMemcpyToSymbol`.
- Write a kernel that accesses the constant memory fields and performs a simple computation on each element of an array: add the offset and multiply by the scale.
- Allocate input and output arrays on both host and device, launch the kernel, copy results back, and print them for verification.
- Include basic error checking for CUDA API calls.
- Ensure the code compiles with `nvcc` and runs on a CUDA-capable device.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Structure to be stored in constant memory
struct Params {
    int  offset;
    float scale;
};

// Declare constant memory variable
__constant__ Params d_params;

// Kernel that uses the constant memory struct
__global__ void processKernel(const int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Read from constant memory
        int outVal = d_in[idx] + d_params.offset;
        d_out[idx] = static_cast<int>(outVal * d_params.scale);
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    const int N = 10;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = i;

    int *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Prepare struct on host and copy to constant memory
    Params h_params;
    h_params.offset = 5;
    h_params.scale  = 2.0f;
    CHECK_CUDA(cudaMemcpyToSymbol(d_params, &h_params, sizeof(Params)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    processKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; ++i) printf("%d ", h_in[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
