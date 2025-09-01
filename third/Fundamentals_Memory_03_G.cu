```c
/*
After a DtoD copy, modify d_a. Does d_b change? (No, it's a deep copy).

The aim of this program is to illustrate that when a device-to-device copy
is performed using cudaMemcpy with cudaMemcpyDeviceToDevice, the destination
buffer receives a deep copy of the source. Consequently, subsequent changes
to the source buffer do not affect the destination buffer. 

To demonstrate this, we:
1. Allocate two device arrays `d_a` and `d_b`.
2. Initialize `d_a` with a known pattern via a kernel.
3. Perform a device-to-device copy from `d_a` to `d_b` using
   `cudaMemcpy(..., cudaMemcpyDeviceToDevice)`.
4. Launch another kernel that modifies `d_a` (e.g., adds a constant to each element).
5. Copy both arrays back to the host and print them.

The output should show that `d_a` has been modified, whereas `d_b`
remains unchanged, confirming that the copy was indeed a deep copy.

The program uses a small array of 10 elements for clarity and
includes basic CUDA error checking.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(err)                                            \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error: %s (err_num=%d) at %s:%d\n",  \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                        \
    }

// Kernel to initialize array with a sequence 0,1,2,...
__global__ void init_array(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = idx;
    }
}

// Kernel to add a constant to each element
__global__ void modify_array(int *arr, int n, int delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] += delta;
    }
}

int main(void) {
    const int N = 10;
    const size_t bytes = N * sizeof(int);

    int *d_a = NULL;
    int *d_b = NULL;
    int h_a[N];
    int h_b[N];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));

    // Initialize d_a
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_array<<<blocks, threadsPerBlock>>>(d_a, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform device-to-device copy: d_b = d_a
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // Modify d_a: add 100 to each element
    modify_array<<<blocks, threadsPerBlock>>>(d_a, N, 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("After modifying d_a:\n");
    printf("h_a: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_a[i]);
    printf("\n");

    printf("h_b (should be unchanged): ");
    for (int i = 0; i < N; ++i) printf("%d ", h_b[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```