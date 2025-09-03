/*
Aim of the program: Write a kernel that modifies the data on the device (e.g., doubles every element) before you copy it back. Verify the result is as expected.

Thinking:
The task is to write a complete CUDA C program that demonstrates the typical pattern of allocating memory on the host, transferring it to the device, launching a kernel that modifies the data, and copying the results back to the host for verification.

Key steps:
1. Define a simple error‑checking macro `checkCudaErrors` that wraps CUDA runtime calls and aborts if an error occurs.
2. Create a host array `h_data` of a reasonable size (e.g., 1024 elements). Initialize it with known values.
3. Allocate device memory `d_data` with `cudaMalloc`.
4. Copy the host data to the device using `cudaMemcpy`.
5. Define a kernel `double_elements` that, given a pointer and a length, doubles each element. Each thread processes one element, using the global index `tid = blockIdx.x * blockDim.x + threadIdx.x`. Guard against out‑of‑bounds accesses.
6. Compute grid and block dimensions such that all elements are covered (e.g., 256 threads per block).
7. Launch the kernel.
8. Copy the modified data back to the host with `cudaMemcpy`.
9. Verify the result by comparing each element to the expected doubled value. Print a message indicating success or failure. Optionally, print the first few elements for manual inspection.
10. Clean up device memory with `cudaFree`.

The code is self‑contained, uses standard CUDA API calls, and compiles with `nvcc`. It prints a confirmation when the verification passes. */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )
inline void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<int>(err), cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

/* Kernel that doubles each element of the array */
__global__ void double_elements(float *d_arr, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_arr[tid] *= 2.0f;
    }
}

int main(void)
{
    const int N = 1024;            // number of elements
    const size_t size = N * sizeof(float);

    // Allocate and initialize host array
    float *h_data = (float *)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);   // simple known values
    }

    // Allocate device array
    float *d_data = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_data, size));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    // Wait for GPU to finish before accessing on host
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy modified data back to host
    checkCudaErrors(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = 2.0f * static_cast<float>(i);
        if (fabs(h_data[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n", i, expected, h_data[i]);
            errors++;
            if (errors > 10) break;   // limit output
        }
    }

    if (errors == 0) {
        printf("All values verified successfully. Kernel doubled each element.\n");
    } else {
        printf("Verification failed with %d errors.\n", errors);
    }

    // Print first 10 results for manual inspection
    printf("First 10 doubled values:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }

    // Clean up
    free(h_data);
    checkCudaErrors(cudaFree(d_data));

    // Reset device (optional)
    checkCudaErrors(cudaDeviceReset());

    return 0;
}
