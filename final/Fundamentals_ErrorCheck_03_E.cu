/*
Why is it good practice to check for errors both immediately after launch and at synchronization? (Catch config errors early, catch execution errors later).

The core idea behind this practice is that CUDA error reporting is asynchronous with respect to the host: a kernel launch itself is just a command that is queued on the device; the actual execution happens later.  
When you launch a kernel, the GPU driver checks the launch configuration (grid/block dimensions, shared memory size, etc.) and will set an error status if any of these parameters are invalid.  Since this check happens synchronously, you can catch it right after the launch by calling cudaGetLastError().  This lets you detect problems like launching a grid with too many blocks, requesting more shared memory than available, or using a launch configuration that is unsupported on the current device.  These errors are purely configuration errors; they do not involve the kernel code executing on the device.

The actual kernel code runs asynchronously.  Any runtime error that occurs while the kernel executes (illegal memory accesses, division by zero, accessing out‑of‑bounds arrays, etc.) will be recorded in the device’s error status, but it will only be propagated back to the host when you perform a synchronous operation, such as cudaDeviceSynchronize() or cudaMemcpy() from device to host.  Therefore, after you have already checked for launch configuration errors, you must still call cudaDeviceSynchronize() (or similar) to flush any execution errors that might have happened during the kernel run.

By checking immediately after the launch you catch configuration issues quickly, and by checking after synchronization you catch execution problems.  Skipping either check can hide bugs: an invalid launch may go unnoticed if you only check later, while a runtime error might be missed if you only check after launch and never synchronize.

The following simple CUDA program demonstrates this two‑step error checking approach: it launches a kernel, checks for launch errors, then synchronizes and checks for execution errors.  The kernel purposely writes to an array, and the error checks illustrate the difference between configuration and execution errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for checking CUDA API calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple kernel that writes to an array
__global__ void simpleKernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Intentionally cause a potential out-of-bounds access if N is too large
        data[idx] = idx;
    }
}

int main(void)
{
    const int N = 10;            // Size of the array
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *h_data = (int *)malloc(N * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, N * sizeof(int)));

    // Launch kernel
    simpleKernel<<<blocks, threadsPerBlock>>>(d_data, N);

    // Check for launch configuration errors immediately after launch
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Launch error: %s\n", cudaGetErrorString(launchErr));
        // Clean up
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Synchronize and check for execution errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Execution error after synchronize: %s\n", cudaGetErrorString(syncErr));
        // Clean up
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Kernel executed successfully. Data:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_data);
    free(h_data);
    return EXIT_SUCCESS;
}
