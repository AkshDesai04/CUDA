```cuda
/*
Aim: Add robust error checking around every CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).

Thinking:
The goal is to wrap each CUDA runtime API call with error checking so that any failure is reported
with a clear message and the program exits gracefully.  A convenient approach is to define a
macro that executes the call, captures the returned cudaError_t, and if it is not cudaSuccess,
prints the file, line, and error string and terminates the program.  We also need to check for
kernel launch errors using cudaGetLastError() and cudaDeviceSynchronize() after the kernel
launch.  All memory allocations (host and device), copies, and frees will use this macro.  The
example program below allocates a small array, copies it to the device, runs a simple kernel
that increments each element, copies the result back, prints it, and finally frees the
resources, with robust error handling at each step.  This demonstrates the requested robust
error checking in a self‑contained CUDA C program.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA runtime API calls */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Simple kernel that adds 1 to each element of the array */
__global__ void add_one(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void)
{
    const int N = 10;
    size_t size = N * sizeof(int);
    int *h_arr = (int *)malloc(size);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    /* Device memory allocation with error checking */
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, size));

    /* Copy data from host to device with error checking */
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    /* Launch kernel and check for launch errors */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host with error checking */
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    /* Print the result */
    printf("Result after adding 1 to each element:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Free device memory with error checking */
    CUDA_CHECK(cudaFree(d_arr));

    /* Free host memory */
    free(h_arr);

    return EXIT_SUCCESS;
}
```