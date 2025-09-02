```cpp
/*
Aim of the program: After a DtoD copy, modify `d_a`. Does `d_b` change? (No, it's a deep copy).

Thinking:
The task is to demonstrate that a device-to-device (DtoD) memcpy creates an independent copy of the data on the GPU. 
After copying `d_a` to `d_b` using `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` flag, we modify the contents of `d_a` 
and then verify that `d_b` remains unchanged. This confirms that the copy is deep, not a shallow reference.

To implement this we:
1. Allocate two device buffers `d_a` and `d_b`.
2. Initialize `d_a` with known values on the host and copy them to the device.
3. Perform a DtoD copy from `d_a` to `d_b`.
4. Modify `d_a` by writing new values to it (again via a host copy for simplicity).
5. Copy both `d_a` and `d_b` back to the host and print their contents.
6. Observe that `d_b` still contains the original values while `d_a` reflects the modifications.

The program is self‑contained, uses CUDA runtime API functions, and includes basic error checking.
It can be compiled with `nvcc deep_copy_demo.cu -o deep_copy_demo` and run on a system with CUDA installed.
*/
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Simple macro for error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    int h_a[N], h_b[N], h_a_host[N], h_b_host[N];
    int *d_a = NULL, *d_b = NULL;

    // Initialize host array h_a with values 0..N-1
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Copy h_a to d_a
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Device-to-Device copy: d_b = d_a
    CHECK_CUDA(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    // Modify d_a: write new values (e.g., set all elements to 100)
    for (int i = 0; i < N; ++i) {
        h_a_host[i] = 100 + i;   // new values
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_a_host, N * sizeof(int), cudaMemcpyHostToDevice));

    // Copy d_a and d_b back to host
    CHECK_CUDA(cudaMemcpy(h_a_host, d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_host, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("After modifying d_a:\n");
    printf("d_a contents: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_a_host[i]);
    }
    printf("\n");

    printf("d_b contents (should be unchanged): ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b_host[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```