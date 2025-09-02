```cpp
/*
Aim of the program:
Implement scalar multiplication: `output[i] = input[i] * scalar_value`.

Thinking:
- We create a CUDA kernel that receives an input array, an output array,
  the scalar value, and the number of elements. Each thread processes
  one array element if the global index is within bounds.
- The host code allocates host memory, initializes an example input array,
  allocates device memory, copies data to the device, launches the kernel
  with a suitable block and grid size, copies the result back, and
  finally cleans up resources.
- Basic error checking is performed on all CUDA API calls using a
  helper macro and function.
- The program prints a few elements of the result to verify correctness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking helper */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel: element-wise scalar multiplication */
__global__ void scalarMul(const float *input, float *output, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] * scalar;
    }
}

int main(void)
{
    /* Problem size */
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    /* Host memory allocation */
    float *h_input  = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    if (h_input == NULL || h_output == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input data */
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)i;   // Example data: 0, 1, 2, ...

    const float scalar = 2.5f;   // Example scalar value

    /* Device memory allocation */
    float *d_input  = NULL;
    float *d_output = NULL;
    cudaCheckError( cudaMalloc((void**)&d_input,  bytes) );
    cudaCheckError( cudaMalloc((void**)&d_output, bytes) );

    /* Copy input from host to device */
    cudaCheckError( cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice) );

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    scalarMul<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, scalar, N);
    cudaCheckError( cudaGetLastError() );          // Check kernel launch
    cudaCheckError( cudaDeviceSynchronize() );     // Wait for completion

    /* Copy result back to host */
    cudaCheckError( cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost) );

    /* Verify a few results */
    printf("Scalar multiplication results (first 10 elements):\n");
    for (int i = 0; i < 10; ++i)
        printf("output[%d] = %.2f\n", i, h_output[i]);

    /* Clean up */
    cudaCheckError( cudaFree(d_input) );
    cudaCheckError( cudaFree(d_output) );
    free(h_input);
    free(h_output);

    return EXIT_SUCCESS;
}
```