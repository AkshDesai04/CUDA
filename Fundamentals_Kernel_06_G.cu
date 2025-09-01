```cpp
/*
 * Aim of the program: Launch the kernel with 10 threads, but have all 10 threads try to write to `data[0]`. 
 * What value do you expect to see when you copy it back? (This is a race condition).
 *
 * Thinking:
 * The kernel will be launched with a single block of 10 threads (threadIdx.x = 0..9). 
 * Each thread will perform a write to the same memory location `data[0]`. 
 * Because the write order is not deterministic, the final value stored in `data[0]` after kernel execution 
 * depends on which thread finishes last. On many GPUs the last thread to write may be the one with the
 * highest threadIdx.x (i.e., 9), but this is not guaranteed by the CUDA memory model. Hence the value
 * printed by the host is nondeterministic and is a classic example of a race condition.
 *
 * The program below demonstrates this by:
 * 1. Allocating a single integer on the device.
 * 2. Launching 10 threads that each write their threadIdx.x to `data[0]`.
 * 3. Copying the result back to the host and printing it.
 * 
 * Because of the race, the printed value may vary from run to run, and typically will be 9 (the highest
 * thread index) on most runs, but this is not guaranteed.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",\
                    __FILE__, __LINE__, err, cudaGetErrorName(err),     \
                    cudaGetErrorString(err));                          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel where each thread writes its threadIdx.x to the same location */
__global__ void raceKernel(int *data)
{
    /* All 10 threads execute this line */
    *data = threadIdx.x;
}

int main(void)
{
    int h_result = 0;
    int *d_result = NULL;

    /* Allocate device memory for a single integer */
    CHECK_CUDA(cudaMalloc((void**)&d_result, sizeof(int)));

    /* Optional: initialize to zero (not necessary) */
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));

    /* Launch kernel with 10 threads in a single block */
    raceKernel<<<1, 10>>>(d_result);

    /* Wait for kernel to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    /* Print the value retrieved from device */
    printf("Value in data[0] after kernel execution: %d\n", h_result);

    /* Clean up */
    CHECK_CUDA(cudaFree(d_result));

    return 0;
}
```