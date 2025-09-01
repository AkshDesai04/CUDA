```cu
/*
Aim: Instead of a second host array, copy the data back into the original host array.

Thinking:
I want to write a minimal CUDA C program that demonstrates how to avoid using a second host array when copying device results back to host.  
The program will:
1. Allocate a single host array `h_data`.
2. Allocate a device array `d_data`.
3. Initialize `h_data` with some values (e.g., 0..N-1).
4. Copy `h_data` to `d_data`.
5. Run a simple kernel that, for each element, increments its value by 1.
6. Copy the results back into the same `h_data` array, overwriting the original contents.
7. Print the first few elements to verify the operation.

Key points:
- Only one host array is used (`h_data`).
- The copy back step uses `cudaMemcpy` with `cudaMemcpyDeviceToHost` and the destination pointer is `h_data`.
- No second host array is created or used.
- Basic error checking is added for CUDA calls.
- The program uses a single kernel launch for clarity.
- The output demonstrates that the data was successfully processed on the GPU and written back to the original host array.

This fulfills the requirement of avoiding a second host array by overwriting the original data after the device computation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that increments each element by 1 */
__global__ void increment_kernel(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_data[idx] += 1;
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    const int N = 10;                 // Number of elements
    const int size = N * sizeof(int);

    /* Allocate host memory */
    int *h_data = (int *)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    /* Allocate device memory */
    int *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));

    /* Copy host array to device */
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    /* Launch kernel: one thread per element */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    /* Copy the result back into the SAME host array (overwriting original data) */
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    /* Free device memory */
    CUDA_CHECK(cudaFree(d_data));

    /* Print the resulting host array */
    printf("Result after incrementing on GPU:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    /* Free host memory */
    free(h_data);

    return EXIT_SUCCESS;
}
```