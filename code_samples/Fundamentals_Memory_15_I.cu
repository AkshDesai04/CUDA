/*
What happens to the value of a `__device__` variable when the application exits?
When a CUDA application terminates, all device memory allocated by the runtime
is released. A `__device__` variable resides in that device memory, so its
contents are destroyed along with the rest of the allocation. Therefore, the
value of a `__device__` variable is **not** preserved across program runs.
Inside a single program execution, the variable persists across kernel
invocations, but once the process ends, the device memory is freed and the
value is lost. The next run starts with the variable initialized to its
declaration value again.

This program demonstrates:
1. A `__device__` variable `devCounter` that is initialized to 0.
2. A kernel that increments `devCounter`.
3. Host code that launches the kernel, reads back the value, and prints it.
4. The fact that after the program exits, the variable does not retain its
   value – the next execution (if run again) will start from 0.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Device variable that will be shared across kernels in the same process */
__device__ int devCounter = 0;

/* Kernel that increments the device counter */
__global__ void incrementCounter(int inc)
{
    /* Simple atomic increment to avoid race conditions */
    atomicAdd(&devCounter, inc);
}

/* Helper function for error checking */
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    int hostCounter;
    const int increments[] = {1, 5, 10};

    /* Print initial value (should be 0) */
    checkCudaError(cudaMemcpyFromSymbol(&hostCounter, devCounter, sizeof(int),
                                        0, cudaMemcpyDeviceToHost),
                   "Copying devCounter to host (initial)");
    printf("Initial devCounter value: %d\n", hostCounter);

    /* Launch kernels to increment devCounter */
    for (int i = 0; i < 3; ++i)
    {
        incrementCounter<<<1, 1>>>(increments[i]);
        checkCudaError(cudaGetLastError(), "Kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "Kernel sync");

        /* Read back after each launch */
        checkCudaError(cudaMemcpyFromSymbol(&hostCounter, devCounter, sizeof(int),
                                            0, cudaMemcpyDeviceToHost),
                       "Copying devCounter to host (after kernel)");
        printf("devCounter after increment %d: %d\n", increments[i], hostCounter);
    }

    /* Clean up and exit */
    printf("Program exiting. devCounter will be destroyed on the device.\n");
    return 0;
}
