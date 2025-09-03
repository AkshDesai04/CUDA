```cuda
/*
Aim of the program:
Declare an array using __device__, e.g., __device__ float lookup_table[256];
Initialize it from the host and use it in a kernel.

Thinking process:
1. Declare a device array of 256 floats: __device__ float lookup_table[256];
2. On the host, create a corresponding array and fill it with some values.
3. Copy the host array to the device array using cudaMemcpyToSymbol.
   This is the appropriate API for copying data to a symbol defined
   with __device__.
4. Write a simple kernel that reads from this lookup table.
   The kernel will take an input array, look up a value in the table
   (for example, use the integer part of the input value modulo 256)
   and write a result to an output array.
5. Allocate host input data, device buffers, launch the kernel, and
   copy the results back to verify that the device array was used
   correctly.
6. Add error checking for CUDA API calls and print some output
   to demonstrate the computation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Device lookup table declaration */
__device__ float lookup_table[256];

/* Kernel that uses the device lookup table */
__global__ void use_lookup(const float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Simple example: use the integer part of the input value
        // modulo 256 to index into the lookup table
        int lut_idx = ((int)input[idx]) & 255; // same as % 256
        output[idx] = input[idx] * lookup_table[lut_idx];
    }
}

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main(void)
{
    const int N = 10;                     // number of elements
    const int LOOKUP_SIZE = 256;          // size of the lookup table

    /* Host lookup table initialization */
    float h_lookup[LOOKUP_SIZE];
    for (int i = 0; i < LOOKUP_SIZE; ++i)
    {
        h_lookup[i] = 1.0f + (float)i;  // simple increasing values
    }

    /* Copy host lookup table to device symbol */
    CHECK_CUDA(cudaMemcpyToSymbol(lookup_table, h_lookup,
                                  LOOKUP_SIZE * sizeof(float),
                                  0, cudaMemcpyHostToDevice));

    /* Host input data */
    float h_input[N];
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = (float)i + 0.5f;   // values 0.5, 1.5, ...
    }

    /* Device buffers */
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    /* Copy input data to device */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    use_lookup<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Copy results back to host */
    float h_output[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Input\tLookupIndex\tLookupValue\tOutput\n");
    for (int i = 0; i < N; ++i)
    {
        int lut_idx = ((int)h_input[i]) & 255;
        printf("%6.2f\t%9d\t%9.2f\t%6.2f\n",
               h_input[i], lut_idx, h_lookup[lut_idx], h_output[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```