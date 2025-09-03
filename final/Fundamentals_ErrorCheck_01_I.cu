```cuda
/*
Aim of the program: 
Write the error checking logic in a helper function `void check_error(cudaError_t code, const char* file, int line)` and have the macro call this function.

Thinking process:
- In CUDA programming, every CUDA runtime API call returns a cudaError_t which indicates success or a particular error.
- It's good practice to check the return value immediately after each call to detect failures early.
- We create a helper function `check_error` that accepts the error code, source file, and line number.
- If the error code is not cudaSuccess, the function prints a detailed error message (including the human‑readable error string, file name, and line number) and terminates the program.
- A macro `CUDA_CHECK` is defined to wrap the helper, passing `__FILE__` and `__LINE__` automatically.
- The macro is used on all CUDA runtime calls: memory allocation, memory copy, kernel launch, synchronization, etc.
- After launching a kernel we also query for errors using `cudaGetLastError()` and pass that to the macro.
- The program demonstrates a simple vector addition kernel that uses this error checking framework.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper function that prints an error message and exits if an error occurs */
void check_error(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error [%d]: %s at %s:%d\n",
                code, cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

/* Macro to automatically pass file and line information */
#define CUDA_CHECK(code) check_error((code), __FILE__, __LINE__)

/* Simple kernel that adds two vectors */
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());
    /* Wait for completion */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("Verification results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("h_c[%d] = %f (expected %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
```