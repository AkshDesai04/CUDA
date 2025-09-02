/*
Try using `__syncthreads()` inside a loop where some threads might exit the loop earlier than others.
This will also cause a deadlock.

--- Thinking and design notes ---

The goal of this program is to illustrate a classic CUDA synchronization
deadlock: a kernel contains a __syncthreads() call inside a loop
condition that allows some threads to exit the loop earlier than
others. Since __syncthreads() requires *all* participating threads in
the block to reach it, a thread that breaks out of the loop will
skip the barrier, leaving the remaining threads stuck waiting forever.
In practice this results in the kernel never finishing and the host
program hanging at cudaDeviceSynchronize().

Implementation outline:

1. Host code allocates an array of integers on the device and fills it
   with values that cause some threads to break out of the loop early.
2. The kernel (deadlockKernel) uses a while-loop that checks a condition
   based on the thread’s array element. If the element is below a
   threshold (e.g. < 5), the thread breaks out of the loop. All
   threads perform some dummy work and then call __syncthreads() before
   incrementing the loop counter.
3. Because some threads break out, those threads skip the __syncthreads()
   call, and the remaining threads are blocked forever at the barrier,
   producing a deadlock.

The code below is deliberately simple and intentionally triggers
the deadlock when run. The host will hang at cudaDeviceSynchronize(),
demonstrating the synchronization issue. Compile with nvcc and run to
observe the hang. The deadlock can be avoided by ensuring all threads
follow the same execution path or by restructuring the loop so that
__syncthreads() is only called when all threads are guaranteed to be
present.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__global__ void deadlockKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int i = 0;
    while (i < 10)
    {
        /* Threads with values less than 5 will exit the loop early. */
        if (d_arr[idx] < 5)
            break;          // This thread skips the barrier below

        /* Dummy computation */
        d_arr[idx] += i;

        /* Barrier inside the loop causes deadlock if some threads break */
        __syncthreads();

        i++;
    }
}

int main(void)
{
    const int N = 64;                 // Number of elements
    const int BATCH = 64;             // Threads per block
    const int BLOCKS = 1;             // Number of blocks

    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array with values 0-9 */
    for (int i = 0; i < N; ++i) {
        h_arr[i] = rand() % 10;   // Random values 0-9
    }

    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Launch kernel that will deadlock for some threads */
    deadlockKernel<<<BLOCKS, BATCH>>>(d_arr, N);

    /* Synchronize to catch errors; this call will hang due to deadlock */
    printf("Launching kernel...\n");
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n",
                cudaGetErrorString(sync_err));
    } else {
        printf("Kernel finished successfully (unexpected)\n");
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_arr));
    free(h_arr);
    return 0;
}
