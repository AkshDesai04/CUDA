```cpp
/*
Aim: Use cudaMemset to initialize d_a to a value (e.g., 5) instead of copying from the host. Then perform the DtoD copy.

Thinking:
The original program copied data from a host array to device memory using cudaMemcpy, and then performed a device‑to‑device copy.  The task is to replace the host copy with a cudaMemset that sets every byte of d_a to a specific value (here 5).  cudaMemset operates on bytes, so setting each byte to 5 results in each integer element being 0x05050505 (84215045 in decimal).  After initializing d_a with cudaMemset, we perform a device‑to‑device copy to d_b, copy the result back to the host, and print a few elements to verify the operation.  Error checking is added for all CUDA API calls to aid debugging and ensure correct execution.  This demonstrates the use of cudaMemset for device memory initialization and a device‑to‑device memcpy.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    const int N = 10;                     // number of elements
    const size_t size = N * sizeof(int);  // size in bytes

    int *h_a = (int*)malloc(size);
    if (!h_a) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // For demonstration, we could initialize h_a, but it will not be used
    for (int i = 0; i < N; ++i) h_a[i] = i;

    int *d_a = nullptr;
    int *d_b = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));

    // Initialize d_a with cudaMemset: set every byte to 5
    CUDA_CHECK(cudaMemset(d_a, 5, size));

    // Perform device-to-device copy from d_a to d_b
    CUDA_CHECK(cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice));

    // Copy result back to host
    int *h_b = (int*)malloc(size);
    if (!h_b) {
        fprintf(stderr, "Failed to allocate host memory for h_b.\n");
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        free(h_a);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print first 10 elements to verify
    printf("Values in h_b after DtoD copy (each element should be 84215045):\n");
    for (int i = 0; i < N; ++i) {
        printf("h_b[%d] = %d\n", i, h_b[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return EXIT_SUCCESS;
}
```