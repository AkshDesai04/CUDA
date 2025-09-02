/*
Aim: Place all CUDA calls in your program on a separate line so the CHECK macro can wrap them cleanly.

Thinking:
1. We want to demonstrate a minimal CUDA program that uses a CHECK macro to wrap each CUDA API call. The macro should capture the return status, and on error, print a message and abort.
2. For the macro to wrap the calls cleanly, each CUDA call must appear on its own line. That means no chaining of calls on the same line, no semicolons inside a block, etc. 
3. The program will:
   - Allocate a small array on the host.
   - Allocate a device array.
   - Copy data to device.
   - Run a simple kernel that increments each element.
   - Copy results back to host.
   - Validate the result.
4. All CUDA calls (cudaMalloc, cudaMemcpy, cudaLaunch, cudaDeviceSynchronize, etc.) will be on separate lines.
5. We use `cudaDeviceSynchronize()` after kernel launch for error checking.
6. The macro will be defined as `CHECK(call)` that checks the return value and prints an error if not `cudaSuccess`.
7. The kernel will be very simple: `__global__ void inc(int *d, int n)` that increments each element.
8. We keep the program self‑contained, no external dependencies beyond CUDA and standard headers.
9. The output will be a .cu file containing the code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls */
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

/* Simple kernel that increments each element by 1 */
__global__ void inc_kernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void)
{
    const int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(int);

    /* Host memory allocation */
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    /* Device memory allocation */
    int *d_arr = NULL;
    CHECK(cudaMalloc(&d_arr, bytes));

    /* Copy data from host to device */
    CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    inc_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    /* Synchronize to catch kernel launch errors */
    CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    CHECK(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    /* Validate results */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i + 1) {
            ++errors;
            if (errors < 10) {  // print first few errors
                fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n",
                        i, i + 1, h_arr[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All %d values correctly incremented.\n", N);
    } else {
        printf("Found %d mismatches out of %d elements.\n", errors, N);
    }

    /* Free device and host memory */
    CHECK(cudaFree(d_arr));
    free(h_arr);

    /* Reset device (optional) */
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
