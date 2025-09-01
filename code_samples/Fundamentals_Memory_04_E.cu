```cpp
/*
What is the limitation of `cudaMemset`? (It can only set every byte to the same value). Explain why you can't use it to initialize an array of floats to `1.0f`.

I considered that `cudaMemset` behaves exactly like the host `memset`: it writes a single byte value to every byte of a memory region.  For a float array, each element is 4 bytes.  Setting each byte to the same value 1 (0x01) does not produce the IEEE‑754 bit pattern for 1.0f, which is 0x3f800000.  Thus using `cudaMemset(d_arr, 1, N*sizeof(float))` will not initialize the array to 1.0f, but rather to a very small number (≈1.4e-45) for each element.  The correct way to set a float array to 1.0f is to use a kernel that writes the value 1.0f to each element, or use `cudaMemset` with 0 to set all elements to 0.0f, since 0x00000000 is the bit pattern for 0.0f.  This program demonstrates the incorrect result of `cudaMemset` for floats and shows a proper kernel-based initialization.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void setToOne(float *arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        arr[idx] = 1.0f;
}

int main(void)
{
    const int N = 10;
    float *d_arr;
    float h_arr[N];

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Attempt to initialize to 1.0f using cudaMemset (incorrect)
    err = cudaMemset(d_arr, 1, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy back to host to inspect values
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    printf("Array after cudaMemset(d_arr, 1, ...):\n");
    for (int i = 0; i < N; ++i)
        printf("%f ", h_arr[i]);
    printf("\n");

    // Correctly initialize to 1.0f using a kernel
    // First, set all elements to 0.0f with cudaMemset
    err = cudaMemset(d_arr, 0, N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset to 0 failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Launch kernel to set to 1.0f
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    setToOne<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy after kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    printf("Array after kernel setToOne:\n");
    for (int i = 0; i < N; ++i)
        printf("%f ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
```