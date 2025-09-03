/*
Implement `output[i] = scalar_value - input[i]`. 
Goal: Provide a simple CUDA C program that takes an input array and a scalar, computes the difference on the GPU by subtracting each input element from the scalar, stores in output array. The program demonstrates memory allocation, data transfer, kernel launch, and result retrieval. It includes error checking, and prints some results for verification. The kernel uses a straightforward 1D grid of threads covering the array size.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel that computes output[i] = scalar - input[i] */
__global__ void subtractScalar(const float *input, float *output, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = scalar - input[idx];
    }
}

int main(void)
{
    const int N = 1024;                 // Number of elements
    const float scalar = 5.0f;          // Scalar value to subtract from

    size_t size_bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_input  = (float*)malloc(size_bytes);
    float *h_output = (float*)malloc(size_bytes);
    if (h_input == NULL || h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input array */
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i;           // Example: 0, 1, 2, ...
    }

    /* Allocate device memory */
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size_bytes));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtractScalar<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, N);
    CHECK_CUDA(cudaGetLastError());   // Check for launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("First 10 results of output[i] = scalar - input[i]:\n");
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    /* Clean up */
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return EXIT_SUCCESS;
}
