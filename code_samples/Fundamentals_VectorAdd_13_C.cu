/*
Implement `C[i] = (A[i] > 0) ? A[i] : 0` (ReLU activation function).
The aim of the program is to demonstrate a simple CUDA implementation of the ReLU activation
function applied element‑wise to an array. This program includes:
- A CUDA kernel that applies the ReLU operation to each element of the input array.
- Host code that allocates and initializes input data, copies it to the device,
  launches the kernel, copies the result back, and prints the output.
- Basic error checking for CUDA API calls.
The code is written in standard C/C++ with CUDA extensions, so it can be compiled
with `nvcc -o relu relu.cu`. It is kept minimal but functional, suitable for educational
purposes and as a template for more complex neural‑network workloads.
*/

/*======================================================================
   CUDA ReLU activation example
======================================================================*/
#include <stdio.h>
#include <cuda_runtime.h>

/* CUDA kernel: apply ReLU to each element */
__global__ void relu_kernel(const float *A, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx];
        C[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const int N = 10;          // number of elements
    const int size = N * sizeof(float);

    /* Host memory allocation */
    float h_A[N];
    float h_C[N];

    /* Initialize input array with some negative and positive values */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (i % 3 - 1) * 2.0f;  // values: -2, 0, 2, -2, 0, 2, ...
    }

    /* Device memory allocation */
    float *d_A = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch ReLU kernel */
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    CUDA_CHECK(cudaGetLastError());          // check kernel launch
    CUDA_CHECK(cudaDeviceSynchronize());     // wait for completion

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Input  :");
    for (int i = 0; i < N; ++i) printf(" %6.2f", h_A[i]);
    printf("\nOutput :");
    for (int i = 0; i < N; ++i) printf(" %6.2f", h_C[i]);
    printf("\n");

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
