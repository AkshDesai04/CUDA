```c
/*
Implement element-wise minimum.

Thinking:
To implement element-wise minimum in CUDA, we need a kernel that reads two input arrays
(a and b), compares each pair of corresponding elements, and writes the smaller value
to an output array. The kernel will be launched with enough threads to cover all
elements. We also need host code to allocate memory on the device, copy data,
launch the kernel, copy back the result, and optionally print the results.

Key steps:
1. Define array size (N) and allocate host arrays with sample data.
2. Allocate device memory for the three arrays.
3. Copy host input arrays to device.
4. Launch a kernel with a grid that covers N elements. Inside the kernel, each
   thread computes its global index and, if it is within bounds, writes
   fminf(a[idx], b[idx]) to the output.
5. Copy the result back to host and free device memory.
6. Print a few results for verification.

We use floating point arrays and the CUDA function fminf for the minimum.
Error checking is added via a simple macro.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void elementWiseMin(const float *a, const float *b, float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = fminf(a[idx], b[idx]);
}

int main(void)
{
    const int N = 1024;
    const int SIZE = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(SIZE);
    float *h_b = (float*)malloc(SIZE);
    float *h_out = (float*)malloc(SIZE);

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)(i % 10);          // values 0-9 repeating
        h_b[i] = (float)((9 - i) % 10);    // values 9-0 repeating
    }

    // Allocate device memory
    float *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_a, SIZE));
    CHECK_CUDA(cudaMalloc((void**)&d_b, SIZE));
    CHECK_CUDA(cudaMalloc((void**)&d_out, SIZE));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    elementWiseMin<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost));

    // Print first 20 results for verification
    printf("Index\tA\tB\tmin(A,B)\n");
    for (int i = 0; i < 20; ++i) {
        printf("%4d\t%.1f\t%.1f\t%.1f\n", i, h_a[i], h_b[i], h_out[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_out);

    return 0;
}
```