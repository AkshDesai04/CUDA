```cpp
/*
 * Aim of the program:
 * Place the `CHECK(cudaDeviceSynchronize())` at the very end of `main`. Is this sufficient to know an error occurred? Is it good for debugging *where* it occurred?
 *
 * Thinking:
 * In CUDA programming, errors can occur asynchronously. Kernel launches are asynchronous with respect to the host; any error that occurs during kernel execution will not be reported until a subsequent synchronous CUDA API call that forces a host‑device synchronization or query. Placing a single `CHECK(cudaDeviceSynchronize())` at the very end of `main` will indeed catch any errors that happened during the kernel execution, because `cudaDeviceSynchronize()` blocks until all preceding GPU work is finished and then reports the last error. However, this approach only tells us that *some* error happened, not exactly where. If we had multiple kernels or device calls interleaved, the final `cudaDeviceSynchronize()` would still return the error code of the *last* error that occurred, but we would lose the exact location (which kernel, which line, etc.). For more precise debugging, we would typically check after each kernel launch (or after each device call) using `cudaGetLastError()` or a macro that wraps that call, so we can pinpoint the exact source of failure. The final check is good for a quick sanity test but not for detailed debugging.
 */

#include <cstdio>
#include <cuda_runtime.h>

/* Macro to check CUDA runtime API calls */
#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* A kernel that deliberately writes out of bounds to trigger an error */
__global__ void badKernel(int *d_arr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* Intentionally access out of bounds when idx >= size */
    d_arr[idx] = idx; /* This will write beyond the allocated memory */
}

int main(void)
{
    const int N = 10;                /* Size of array on device */
    int *d_arr = nullptr;

    /* Allocate device memory */
    CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    /* Launch kernel with more threads than array size to trigger an error */
    int threadsPerBlock = 32;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; /* normally enough */
    /* Intentionally launch one more block to exceed bounds */
    blocks += 1;

    /* No error check here: we want to test the final CHECK at the end */
    badKernel<<<blocks, threadsPerBlock>>>(d_arr, N);

    /* The only error check is at the very end of main */
    CHECK(cudaDeviceSynchronize());

    /* If we reach here, everything succeeded (unlikely in this example) */
    printf("Kernel executed successfully.\n");

    /* Clean up */
    CHECK(cudaFree(d_arr));
    return 0;
}
```