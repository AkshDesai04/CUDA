/*
Write a kernel that accepts two integer pointers, `a` and `b`. The kernel (with a single thread) should read the value from `a`, add 10 to it, and write the result to `b`. The host must allocate, initialize `a`, and verify the result in `b`.

The goal of this program is to demonstrate a minimal CUDA kernel that performs a simple arithmetic operation using a single thread. I want to keep the example very small so that the reader can see all the necessary CUDA steps:

1. Allocate memory on the host for two integers (`h_a` and `h_b`).
2. Allocate memory on the device for two integers (`d_a` and `d_b`).
3. Initialize the host input `h_a` with some value (e.g., 5).
4. Copy `h_a` to the device memory `d_a` using `cudaMemcpy`.
5. Launch the kernel with a single thread (`<<<1, 1>>>`) which dereferences `a`, adds 10, and stores the result in `b`.
6. Copy the result from `d_b` back to the host variable `h_b`.
7. Verify that `h_b` equals `h_a + 10` and print the result.
8. Free all allocated memory and check for CUDA errors at each step.

The kernel is trivial: `*b = *a + 10;`. I use pointer dereferencing because the arguments are pointers to single integers. The host code uses `cudaMalloc`/`cudaMemcpy` and standard error checking to make sure everything works. The program is self‑contained and can be compiled with `nvcc`.

Below is the full code with all the steps implemented.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that adds 10 to the value pointed to by `a` and writes the result to `b`. */
__global__ void add10(int *a, int *b)
{
    // Since we launch with one thread, we can safely dereference.
    *b = *a + 10;
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    int h_a = 5;          // Host input
    int h_b = 0;          // Host output

    int *d_a = NULL;      // Device pointer for input
    int *d_b = NULL;      // Device pointer for output

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(int)));

    /* Copy input from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));

    /* Launch kernel with a single thread */
    add10<<<1, 1>>>(d_a, d_b);
    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify result */
    if (h_b == h_a + 10) {
        printf("Success: %d + 10 = %d\n", h_a, h_b);
    } else {
        printf("Failure: Expected %d, got %d\n", h_a + 10, h_b);
    }

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
