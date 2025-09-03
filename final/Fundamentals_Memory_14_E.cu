```cuda
/*
 * Aim of the program:
 * Use `cudaDeviceSynchronize` and proper error checking. An out-of-bounds access often results in `cudaErrorIllegalAddress`. Catch and report this error.
 *
 * Thinking:
 * 1. We want a minimal, self‑contained CUDA program that deliberately triggers an illegal memory access on the device.
 * 2. To do this, we allocate a small array on the device, but launch a kernel with more threads than elements.
 * 3. Each thread writes to its index in the array. Threads with indices beyond the array size will write out of bounds.
 * 4. After the kernel launch we use `cudaGetLastError()` to catch any launch‑time errors.
 * 5. Then we call `cudaDeviceSynchronize()` to force completion of the kernel execution on the device.
 * 6. If the kernel performed an illegal address access, `cudaDeviceSynchronize()` will return `cudaErrorIllegalAddress`.
 * 7. We wrap CUDA API calls in a `CHECK_CUDA()` macro that prints the error string and aborts on failure.
 * 8. The program prints whether an illegal address error was detected or if execution succeeded.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA API call results */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel that intentionally writes out of bounds */
__global__ void write_out_of_bounds(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread writes to its index regardless of bounds */
    data[idx] = idx * 2;
}

int main(void) {
    const int N = 10;              /* Size of the device array */
    const int threadsPerBlock = 32;
    const int blocks = 1;          /* Launch fewer blocks but more threads */

    /* Allocate device memory */
    int *d_array;
    CHECK_CUDA(cudaMalloc((void **)&d_array, N * sizeof(int)));

    /* Launch kernel with more threads than elements */
    write_out_of_bounds<<<blocks, threadsPerBlock>>>(d_array, N);

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    /* Synchronize device and check for illegal address errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        if (err == cudaErrorIllegalAddress) {
            fprintf(stderr, "Detected illegal memory access during kernel execution.\n");
        } else {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        }
    } else {
        printf("Kernel executed successfully (no illegal address detected).\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_array));

    return 0;
}
```