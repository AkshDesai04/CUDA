/*
What is the limitation of `cudaMemset`? (It can only set every byte to the same value). Explain why you can't use it to initialize an array of floats to `1.0f`.

My thinking:
`cudaMemset` is analogous to the standard C `memset` function. It takes a pointer, a value of type `int` that is interpreted as an unsigned byte (0–255), and a size in bytes. It writes that byte value into each of the specified bytes in the target memory region. Because it operates on a per‑byte basis, you cannot specify a multi‑byte pattern (such as the 4‑byte representation of a 32‑bit float). A float value like `1.0f` is represented in IEEE‑754 as the 32‑bit pattern `0x3f800000`. If you tried to use `cudaMemset(d_ptr, 0x3f80, N*sizeof(float))`, the function would only write the low byte `0x80` into each of the 4 bytes, resulting in a pattern `0x80808080` for every element—far from the intended `0x3f800000`. Therefore `cudaMemset` cannot be used to set all elements of a float array to `1.0f`; you need to launch a kernel (or use a host-side helper like `thrust::fill`) to perform element‑wise initialization.
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void setToOne(float* arr, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = 1.0f;
    }
}

int main() {
    const size_t N = 10;
    float* d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Attempt to set all elements to 1.0f using cudaMemset (will fail)
    // Each byte will be set to 0x01, resulting in float value 0x01010101 (~2.94e-39)
    err = cudaMemset(d_arr, 0x01, N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_arr);
        return -1;
    }

    // Copy back to host and print
    float h_arr[N];
    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "After cudaMemset with byte 0x01, first few elements:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    // Now correctly set to 1.0f using a kernel
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    setToOne<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "After kernel setToOne, first few elements:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_arr);
    return 0;
}
