```cuda
/*
Inside the kernel, add a `printf` from a single thread (e.g., `if (global_id == 12345)`) to inspect a value.

Thinking process:

1. Goal is to demonstrate how to use `printf` inside a CUDA kernel and restrict it to only one thread.
2. We choose a simple kernel that increments an array element. 
   The array is large enough (size 20000) so that thread ID 12345 exists.
3. In the kernel, compute the global thread ID.  
   If it equals 12345, use `printf` to output the value of the array element at that index.
4. On the host side:
   - Allocate and initialize an array of ints with values 0..N-1.
   - Allocate device memory and copy the array.
   - Launch kernel with enough blocks/threads to cover all elements.
   - Synchronize and copy the array back.
   - Print a few elements on the host to verify the kernel worked.
5. Use a simple error-checking macro for CUDA API calls.
6. The program is self‑contained and compiles with `nvcc`.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void increment_array(int *d_arr, int N)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) {
        d_arr[global_id] += 1; // simple operation

        // Single-thread printf
        if (global_id == 12345) {
            printf("Thread %d has value %d before increment\n", global_id, d_arr[global_id] - 1);
            printf("Thread %d has value %d after increment\n", global_id, d_arr[global_id]);
        }
    }
}

int main(void)
{
    const int N = 20000;
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    increment_array<<<blocks, threads_per_block>>>(d_arr, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Verify a few values
    printf("Sample values after kernel execution:\n");
    for (int i = 12340; i <= 12350; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    return 0;
}
```