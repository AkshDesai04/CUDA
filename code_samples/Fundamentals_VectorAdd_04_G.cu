```cuda
/* 
Aim of the program: Implement for character arrays: `C[i] = A[i] - B[i]`.

Thought process:
- The goal is to perform an element-wise subtraction of two character arrays A and B, storing the result in array C.
- Since CUDA operates on device memory, we need to allocate memory on the GPU, copy the input arrays to the device, launch a kernel to perform the subtraction, and then copy the result back to the host.
- We will use a simple kernel where each thread handles one element. Boundary checks ensure we don't access out of bounds.
- The subtraction of two `char` values is performed in integer arithmetic to avoid unexpected promotion or overflow, but the result is stored back as a `char` for consistency with the input type.
- For demonstration, we initialize A and B with a small set of values and print the resulting array on the host.
- We include basic error checking for CUDA API calls using a macro for readability.
- The program is written in CUDA C and can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel to subtract two character arrays element-wise */
__global__ void subtractChars(const char* A, const char* B, char* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        /* Perform subtraction in integer space to avoid overflow issues */
        int diff = (int)A[idx] - (int)B[idx];
        C[idx] = (char)diff;
    }
}

int main(void)
{
    /* Define array size */
    const int N = 10;
    size_t size = N * sizeof(char);

    /* Allocate host arrays and initialize with sample values */
    char h_A[N] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' };
    char h_B[N] = { 'J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A' };
    char h_C[N];

    /* Allocate device memory */
    char *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    /* Copy input arrays from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel: one block with N threads (sufficient for small N) */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtractChars<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("A : ");
    for (int i = 0; i < N; i++) printf("%c ", h_A[i]);
    printf("\n");

    printf("B : ");
    for (int i = 0; i < N; i++) printf("%c ", h_B[i]);
    printf("\n");

    printf("C : ");
    for (int i = 0; i < N; i++) {
        /* Print as signed integer to show subtraction result clearly */
        printf("%d ", (int)h_C[i]);
    }
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```