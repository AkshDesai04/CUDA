/*
Implement `output[i] = max(input[i], scalar_value)`.
The goal of this program is to demonstrate how to apply a simple element-wise operation
on a GPU using CUDA.  The operation is a comparison between each element of an
input array and a scalar value; the larger of the two is written to an output
array.  To keep the example straightforward we use single-precision floats,
allocate a small array on the host, copy it to the device, launch a kernel that
performs the max operation, copy the results back, and print them to verify
correctness.

Design decisions:
- We use a 1D grid of 1D blocks; each thread handles one element.
- Error checking is done via a helper macro that aborts if a CUDA call fails.
- The scalar is passed to the kernel as an argument, so the kernel is reusable
  for any scalar value.
- For simplicity the number of threads per block is chosen as 256, a common
  default that works well for many GPUs.  The grid size is computed so that
  all elements are covered.
- We print the results on the host; the kernel itself only performs the
  computation.
- The program is self‑contained and can be compiled with `nvcc -o max_example max_example.cu`.

This example shows how to structure a basic CUDA program: host memory allocation,
device memory allocation, data transfer, kernel launch, result retrieval, and
cleanup.  It can be extended to larger arrays, other data types, or more
complex operations as needed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* Kernel that computes output[i] = max(input[i], scalar) */
__global__ void maxKernel(const float *input, float *output, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = (val > scalar) ? val : scalar;
    }
}

int main(void)
{
    const int N = 10;                      // Size of the array
    const float scalar_value = 5.0f;       // The scalar to compare against

    /* Host memory allocation */
    float h_input[N];
    float h_output[N];

    /* Initialize input with some values */
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i - 5); // Values: -5, -4, ..., 4
    }

    /* Device memory allocation */
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(float)));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    maxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar_value, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Input\t\tMax(Input, %.1f)\n", scalar_value);
    for (int i = 0; i < N; ++i) {
        printf("%6.1f\t\t%6.1f\n", h_input[i], h_output[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
