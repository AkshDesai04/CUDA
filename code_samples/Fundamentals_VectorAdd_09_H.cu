/*
Aim: Pass all scalars and vector pointers to the kernel.

The goal of this program is to demonstrate how to pass multiple scalar values and pointers to vectors into a CUDA kernel.  
To achieve this, the program:
1. Allocates and initializes a host input vector.
2. Copies the input vector to the device.
3. Defines a kernel that receives:
   - A pointer to the input vector.
   - A pointer to an output vector.
   - A scalar multiplier.
   - The length of the vector.
   - An offset value (also a scalar) to show that multiple scalars can be passed.
4. Launches the kernel with a suitable grid and block configuration.
5. Copies the resulting output vector back to the host.
6. Prints the results to verify correct operation.
7. Cleans up all allocated memory.

This simple example shows that any combination of scalars and pointers can be forwarded to a CUDA kernel, and the kernel can use them just like ordinary function arguments.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s:%d : %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that multiplies each element by a scalar.
// Parameters:
//   in     - pointer to input vector on device
//   out    - pointer to output vector on device
//   scalar - scalar multiplier
//   length - number of elements in the vector
//   offset - starting index offset for the kernel threads
__global__ void multiply_kernel(const float *in, float *out, float scalar, int length, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < length)
    {
        out[idx] = in[idx] * scalar;
    }
}

int main(void)
{
    const int length = 10;      // Number of elements in the vectors
    const int offset = 0;       // Starting offset for kernel threads
    const float scalar = 2.5f;  // Scalar multiplier

    // Allocate host memory
    float *h_in  = (float *)malloc(length * sizeof(float));
    float *h_out = (float *)malloc(length * sizeof(float));
    if (h_in == NULL || h_out == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input vector
    for (int i = 0; i < length; ++i)
    {
        h_in[i] = (float)(i + 1);  // 1.0, 2.0, ..., length
    }

    // Allocate device memory
    float *d_in  = NULL;
    float *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_in,  length * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_out, length * sizeof(float)));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, length * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, scalar, length, offset);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, length * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input vector:\n");
    for (int i = 0; i < length; ++i)
    {
        printf("%f ", h_in[i]);
    }
    printf("\n");

    printf("Output vector (after multiplying by %f):\n", scalar);
    for (int i = 0; i < length; ++i)
    {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return EXIT_SUCCESS;
}
