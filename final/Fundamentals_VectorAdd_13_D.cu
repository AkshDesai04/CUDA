/*
Use `fmaxf()` from the CUDA math library instead of the ternary operator.

This program demonstrates how to compute the element‑wise maximum of two float arrays
on the GPU using the CUDA math function `fmaxf`.  The goal is to replace a typical
ternary expression such as
    max = (a > b) ? a : b;
with a more concise and potentially faster library call.
The kernel simply applies `fmaxf` to each pair of elements and writes the result
to an output array.  After launching the kernel, the host code copies the result
back to the CPU and prints the first few values for verification.

Key points covered in this code:
1. Allocation of device memory for two input arrays and one output array.
2. Initialization of the host arrays with sample data.
3. Launching a CUDA kernel that uses `fmaxf`.
4. Copying results back to host memory and printing them.
5. Basic CUDA error checking via the `checkCuda` macro.

The program is self‑contained and can be compiled with a standard CUDA compiler:
    nvcc -o max_demo max_demo.cu
Then run it with:
    ./max_demo
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper macro for CUDA error checking
#define checkCuda(call)                                 \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

// CUDA kernel that computes element‑wise maximum using fmaxf
__global__ void elementwise_max(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(a[idx], b[idx]);  // use fmaxf instead of ternary operator
    }
}

int main(void) {
    const int N = 10;  // size of arrays
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)(i);          // 0.0, 1.0, 2.0, ...
        h_b[i] = (float)(N - i - 1);  // 9.0, 8.0, 7.0, ...
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc((void**)&d_a, size));
    checkCuda(cudaMalloc((void**)&d_b, size));
    checkCuda(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_max<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCuda(cudaGetLastError());  // check for launch errors

    // Copy result back to host
    checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Index\tA\tB\tmax(A,B)\n");
    for (int i = 0; i < N; ++i) {
        printf("%d\t%.2f\t%.2f\t%.2f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
