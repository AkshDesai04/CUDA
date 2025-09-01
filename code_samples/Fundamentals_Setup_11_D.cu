/*
Aim of the program:
Create a second empty kernel with a different name and launch both kernels sequentially from `main`.

Thinking and design decisions:
1. I chose to implement two CUDA kernels. The first kernel (`firstKernel`) writes a simple sequence of numbers into a device array. The second kernel (`secondEmptyKernel`) is intentionally empty and does nothing, just to demonstrate that we can launch multiple kernels in sequence.
2. The device memory allocation is done in the host using `cudaMalloc`. A simple array of 10 integers is allocated, initialized to zero on the host, then copied to the device.
3. After launching the first kernel, I copy the device data back to the host and print the values so that we can see the effect of the first kernel.
4. The second kernel is launched with the same grid/block configuration but performs no operations. This demonstrates sequential kernel launch where the second kernel does not alter any data.
5. Error checking is performed after each CUDA call using the `cudaError_t` return value. If an error occurs, the program prints the error string and exits.
6. I used the standard CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, etc.) for simplicity.
7. The code is fully self-contained and can be compiled with `nvcc` to produce an executable. No external dependencies are required.
8. Comments are added to clarify each step for readability and maintainability.

Now the code follows below:
*/

#include <stdio.h>
#include <stdlib.h>

__global__ void firstKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx * 10; // simple operation: store multiples of 10
    }
}

__global__ void secondEmptyKernel(int *d_arr, int n) {
    // Intentionally left empty. This kernel performs no operations.
    // It is included to demonstrate launching multiple kernels sequentially.
    // The parameters are kept for API consistency.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // No action performed
    }
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(int);

    int *h_arr = (int *)malloc(size);
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array to zero
    for (int i = 0; i < N; ++i) {
        h_arr[i] = 0;
    }

    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy host data to device (currently all zeros)
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch first kernel
    firstKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "firstKernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "firstKernel synchronization failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy result back to host and print
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H after firstKernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    printf("Result after firstKernel:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Launch second empty kernel
    secondEmptyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "secondEmptyKernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "secondEmptyKernel synchronization failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy data again to host (should be unchanged)
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H after secondEmptyKernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    printf("Result after secondEmptyKernel (should be unchanged):\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
