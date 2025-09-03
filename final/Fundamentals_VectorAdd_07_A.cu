/*
Aim: Implement a kernel that cubes every element: `output[i] = input[i] * input[i] * input[i]`.

Thinking:
To accomplish this, we will write a simple CUDA program that:
1. Defines a device kernel `cubeKernel` that accepts an input array, an output array, and the number of elements.
   Each thread computes its global index `idx` and, if within bounds, reads `input[idx]`, cubes it by multiplying
   it three times, and writes the result to `output[idx]`.
2. In `main`, we:
   - Allocate a host array of floats and initialize it with sample values.
   - Allocate device memory for input and output arrays.
   - Copy the input data from host to device.
   - Determine a reasonable block and grid size (e.g., 256 threads per block).
   - Launch the kernel.
   - Copy the results back from device to host.
   - Print a few results to verify correctness.
   - Free all allocated memory.
3. We include basic error checking after CUDA API calls using a helper macro `CHECK_CUDA` to make the code robust.
4. The program is self‑contained, uses standard CUDA headers, and can be compiled with `nvcc` to produce an executable.
5. No external files or libraries are required, and the output will be printed to the console.

The program demonstrates the requested kernel and provides a complete, compilable CUDA example.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for checking CUDA errors */
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* CUDA kernel that cubes each element of the input array */
__global__ void cubeKernel(const float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = val * val * val;  // cube the value
    }
}

int main(void)
{
    const int N = 1024;               // number of elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input with some values */
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i) + 1.0f;  // simple pattern
    }

    /* Allocate device memory */
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the kernel */
    cubeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());   // check for launch errors

    /* Copy the result back to host */
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    /* Verify results by printing first 10 elements */
    printf("First 10 results of cubing input values:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("input[%d] = %f, output[%d] = %f\n", i, h_input[i], i, h_output[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}
