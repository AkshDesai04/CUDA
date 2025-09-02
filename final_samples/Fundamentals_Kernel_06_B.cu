/*
Aim: Modify the kernel to write the value of its `threadIdx.x` to `data[0]`. Launch with one thread and verify.

Thinking:
- The goal is to create a minimal CUDA program that demonstrates writing the thread index to a global memory location.
- The kernel will simply store `threadIdx.x` into the first element of the passed array.
- We will launch the kernel with a single block containing a single thread (`<<<1,1>>>`) to satisfy the requirement of "one thread".
- On the host side, we allocate device memory for an `int`, launch the kernel, copy the result back, and print it to verify.
- Error checking is included to catch any launch or memory issues.
- The program uses standard C headers and CUDA runtime API, making it portable and compilable with `nvcc`.
- After printing the result, we free the device memory and return.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that writes its threadIdx.x to data[0] */
__global__ void writeThreadIdx(int *data) {
    // Since we launch with one thread, threadIdx.x will be 0.
    data[0] = threadIdx.x;
}

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    int *d_data = NULL;
    int h_data = 0;

    /* Allocate device memory for one integer */
    CUDA_CHECK(cudaMalloc((void **)&d_data, sizeof(int)));

    /* Launch kernel with one block of one thread */
    writeThreadIdx<<<1, 1>>>(d_data);
    CUDA_CHECK(cudaGetLastError()); /* Check for launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify and print the result */
    printf("ThreadIdx.x written to data[0] = %d\n", h_data);

    /* Clean up */
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
