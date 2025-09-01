/*
Create a C/C++ preprocessor macro `CHECK(call)` that wraps a CUDA API call. The macro should check the returned `cudaError_t` value. If it's not `cudaSuccess`, it should print the error code, the error string from `cudaGetErrorString`, the file name (`__FILE__`), and the line number (`__LINE__`), and then exit the program.

Thinking process:
1. The macro needs to evaluate the CUDA call once and capture its return value.  
2. If the return value is not `cudaSuccess`, we should use `fprintf` to write the error information to `stderr`.  
3. We include the error code, the human‑readable string via `cudaGetErrorString`, and the context (`__FILE__` and `__LINE__`).  
4. After reporting, the program must terminate, so we use `exit(EXIT_FAILURE)`.  
5. To make the macro safe to use as a single statement, wrap it in a `do { ... } while(0)` construct.  
6. For demonstration, the program will allocate a small array on the GPU, launch a dummy kernel, and free the memory, using `CHECK` after each CUDA API call.  
7. All necessary headers are included: `<stdio.h>`, `<stdlib.h>`, and `<cuda_runtime.h>`.  
8. The kernel does nothing; its purpose is only to show that `CHECK` can be used after a kernel launch.  
9. The program ends with a clean return value of 0 if all CUDA calls succeed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CHECK(call)                                           \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error: %d: %s. %s:%d\n",    \
                    err, cudaGetErrorString(err),             \
                    __FILE__, __LINE__);                      \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

/* Dummy kernel that does nothing */
__global__ void dummyKernel(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f; /* just a trivial operation */
    }
}

int main(void)
{
    const int N = 256;
    float *d_data = NULL;

    /* Allocate device memory */
    CHECK(cudaMalloc((void**)&d_data, N * sizeof(float)));

    /* Launch kernel */
    dummyKernel<<<(N + 255) / 256, 256>>>(d_data, N);

    /* Check for kernel launch errors */
    CHECK(cudaGetLastError());

    /* Synchronize device to catch any runtime errors */
    CHECK(cudaDeviceSynchronize());

    /* Free device memory */
    CHECK(cudaFree(d_data));

    /* All good */
    printf("All CUDA calls succeeded.\n");
    return 0;
}
