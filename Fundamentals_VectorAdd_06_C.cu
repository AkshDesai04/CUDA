```c
/*
Aim of the program:
Implement DAXPY, the double-precision version of the operation. This program performs the operation y = a * x + y on two double precision arrays using CUDA. It demonstrates memory allocation on the host and device, data transfer between host and device, kernel execution, and result verification.

Thoughts:
- The kernel should launch enough threads to cover all elements of the arrays; use a block size of 256 threads and compute the grid size accordingly.
- Use proper bounds checking inside the kernel to avoid out-of-bounds memory accesses.
- Include a simple error-checking macro for CUDA API calls to make debugging easier.
- Allocate memory on the host for arrays x and y, initialize them with sample values.
- Allocate memory on the device for x_d and y_d, copy data from host to device.
- Launch the kernel and then copy the result y_d back to host memory.
- Verify a few elements of the result to ensure correctness.
- Print out a few sample values before and after the operation for visual confirmation.
- The program is self-contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// CUDA kernel for DAXPY: y = a * x + y
__global__ void daxpy_kernel(double a, const double *x, double *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // Size of arrays (1M elements)
    const double a = 2.5;           // Scalar multiplier

    // Host memory allocation
    double *h_x = (double *)malloc(N * sizeof(double));
    double *h_y = (double *)malloc(N * sizeof(double));
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_x[i] = (double)i;            // x[i] = i
        h_y[i] = 1.0;                  // y[i] = 1.0
    }

    // Device memory allocation
    double *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, N * sizeof(double)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the DAXPY kernel
    daxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());      // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (int i = 0; i < 5; ++i) {
        double expected = a * (double)i + 1.0;
        printf("y[%d] = %f (expected %f)\n", i, h_y[i], expected);
    }

    // Clean up
    free(h_x);
    free(h_y);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
```