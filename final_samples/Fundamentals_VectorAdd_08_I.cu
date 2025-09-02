/*
Pass the scalar via constant memory.

The goal of this program is to demonstrate how to pass a scalar value to a CUDA kernel using the constant memory space. Constant memory is a read‑only memory region on the GPU that is cached and is ideal for values that are accessed by all threads and do not change during kernel execution. By placing a scalar in constant memory we avoid redundant global memory accesses and allow all threads to read the same value efficiently.

Design choices:
1. A simple vector multiplication kernel is used. Each thread multiplies an element of an input array by the scalar stored in constant memory and writes the result to an output array.
2. The scalar is declared as a global variable in constant memory: `__constant__ float d_scalar;`.
3. On the host side, the scalar value is copied into constant memory using `cudaMemcpyToSymbol`.
4. The program uses standard CUDA runtime API calls with basic error checking.
5. For demonstration purposes, the host allocates a small array, initializes it, runs the kernel, copies back the results, and prints them.

This code is self‑contained and can be compiled with `nvcc`:
    nvcc -o constant_scalar constant_scalar.cu
and executed:
    ./constant_scalar
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

/* ---------- Constant memory declaration ---------- */
__constant__ float d_scalar;

/* ---------- CUDA kernel: multiply each element by scalar ---------- */
__global__ void scaleArray(float *out, const float *in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = in[idx] * d_scalar;  // read from constant memory
    }
}

/* ---------- Simple CUDA error checking macro ---------- */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main()
{
    /* ---------- Host data ---------- */
    const int N = 16;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = static_cast<float>(i);   // 0, 1, 2, ...
    }
    float scalar = 2.5f;   // scalar value to be passed via constant memory

    /* ---------- Device memory allocation ---------- */
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    /* ---------- Copy input array to device ---------- */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    /* ---------- Copy scalar to constant memory ---------- */
    CUDA_CHECK(cudaMemcpyToSymbol(d_scalar, &scalar, sizeof(float), 0, cudaMemcpyHostToDevice));

    /* ---------- Launch kernel ---------- */
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    scaleArray<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());  // check for launch errors

    /* ---------- Copy result back to host ---------- */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* ---------- Print results ---------- */
    std::cout << "Input array:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_in[i] << ' ';
    std::cout << "\n\nScaled array (each element multiplied by " << scalar << "):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_out[i] << ' ';
    std::cout << "\n";

    /* ---------- Clean up ---------- */
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
