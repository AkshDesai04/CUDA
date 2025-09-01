/*
Implement DAXPY, the double-precision version of the operation.

Thinking:
The task is to provide a CUDA C program that performs the DAXPY operation:
    y = a * x + y
for double-precision arrays.  
We will create a simple kernel that executes this operation element-wise.  
Host code will:
1. Allocate host memory for input vectors x and y, and a scalar a.
2. Initialize x and y with some test values.
3. Allocate device memory, copy host data to device.
4. Launch the kernel with an appropriate grid/block configuration.
5. Copy the result back to host and print a few values for verification.
6. Clean up resources.

We will include error checking after CUDA API calls to catch any failures.  
The program will be self‑contained in a single .cu file, ready to compile with nvcc.  
All comments and explanation are provided above this code block.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Kernel for DAXPY: y = a * x + y
__global__ void daxpy_kernel(double a, const double* x, double* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // Size of vectors (1M elements)
    const double a = 2.5;           // Scalar multiplier

    // Host allocations
    double *h_x = (double*)malloc(N * sizeof(double));
    double *h_y = (double*)malloc(N * sizeof(double));
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<double>(i) + 1.0;   // x[i] = i + 1
        h_y[i] = static_cast<double>(i) + 0.5;   // y[i] = i + 0.5
    }

    // Device allocations
    double *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the DAXPY kernel
    daxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_x, d_y, N);
    CHECK_CUDA(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Print a few results for verification
    printf("DAXPY results (first 10 elements):\n");
    for (int i = 0; i < 10; ++i) {
        double expected = a * (static_cast<double>(i) + 1.0) + (static_cast<double>(i) + 0.5);
        printf("y[%d] = %f (expected %f)\n", i, h_y[i], expected);
    }

    // Clean up
    free(h_x);
    free(h_y);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return EXIT_SUCCESS;
}
