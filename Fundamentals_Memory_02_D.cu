```cpp
/* 
Aim of the program: Copy only the first half of the array back from the device and verify it.

Thinking and plan:

1. Define the problem size `N` (e.g., 1024) and compute `half = N/2`.
2. Allocate a host array `h_data` of size `N` and initialize it with some known values (e.g., `h_data[i] = i`).
3. Allocate a device array `d_data` of size `N`.
4. Copy the entire host array to the device (`cudaMemcpy`).
5. Launch a simple kernel that performs a deterministic operation on each element of `d_data` (e.g., add 10 to each element).
6. After kernel completion, copy **only** the first half of the device array back to the host:
   - Use `cudaMemcpy` with a destination pointer `h_data` (overwrites the first half) and a source pointer offset `d_data + half`.
   - Specify the byte count as `half * sizeof(int)`.
7. Verify that the first half of `h_data` now contains the expected transformed values (`i + 10`), while the second half remains unchanged.
8. Print a success message if verification passes, otherwise report mismatches.
9. Clean up all allocated memory and handle any CUDA errors with a simple macro.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",                \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

__global__ void addTen(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] += 10;
}

int main(void)
{
    const int N = 1024;
    const int half = N / 2;

    // Host memory allocation
    int *h_data = (int *)malloc(N * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Device memory allocation
    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(int)));

    // Copy entire array from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel to add 10 to each element
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTen<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy only the first half back to host
    CHECK_CUDA(cudaMemcpy(h_data,
                          d_data,                     // source pointer
                          half * sizeof(int),         // size to copy
                          cudaMemcpyDeviceToHost));

    // Verification
    bool success = true;
    for (int i = 0; i < half; ++i) {
        if (h_data[i] != i + 10) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, i + 10, h_data[i]);
            success = false;
            break;
        }
    }

    // Optionally check that the second half remains unchanged
    for (int i = half; i < N; ++i) {
        if (h_data[i] != i) {
            fprintf(stderr, "Second half altered at index %d: expected %d, got %d\n",
                    i, i, h_data[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Verification succeeded: first half updated correctly, second half unchanged.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```