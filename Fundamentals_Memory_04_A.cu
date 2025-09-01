/*
Write a `main` function that calls your function, then copies the zeroed array back to the host and verifies that all elements are indeed 0.

Thinking:
- We need to allocate a host array and a device array.
- The kernel will set each element to 0.
- After kernel launch, copy device array back to host.
- Verify by looping through host array and checking all zeros.
- Use simple kernel: each thread writes 0 to its index.
- Use error checking for CUDA API calls.
- Set array size (e.g., 1024 elements).
- Use 256 threads per block, compute number of blocks accordingly.
- Provide debug prints if mismatch found.
- Ensure correct memory allocation and freeing.
- Use cudaMemcpy for host to device and device to host.
- Use cudaDeviceSynchronize after kernel launch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void zeroArray(float *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = 0.0f;
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);
    float *h_arr = (float*)malloc(size);
    float *d_arr = NULL;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    zeroArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }

    // Copy back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return -1;
    }

    // Verify all zeros
    int allZero = 1;
    for (int i = 0; i < n; ++i) {
        if (h_arr[i] != 0.0f) {
            printf("Element %d is not zero: %f\n", i, h_arr[i]);
            allZero = 0;
            break;
        }
    }
    if (allZero) {
        printf("All elements are zero.\n");
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);
    cudaDeviceReset();

    return 0;
}