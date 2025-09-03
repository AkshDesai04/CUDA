```cuda
/*
Perform an out-of-bounds read instead of a write and analyze the `cuda-memcheck` report.

Thinking Process:
1. Goal: deliberately cause a device-side out-of-bounds read so that cuda-memcheck can detect it.
2. Strategy:
   - Allocate a device array of a known size (e.g., N = 10).
   - Launch a simple kernel that reads from an index that is one past the last valid index (index N).
   - Store that value into another device variable (result) so we can copy it back and observe it.
   - No device memory is written out-of-bounds; only a read operation is performed.
3. Implementation Details:
   - Use `cudaMalloc` to allocate device memory for the array and for the result.
   - Use `cudaMemcpy` to copy the result back to the host for printing.
   - The kernel will be launched with a single thread so that the read operation is deterministic.
   - We will check for CUDA errors after each API call for robustness.
4. Expected Behavior:
   - When running normally, the program will print the value read from the out-of-bounds location (often 0 or garbage).
   - When running under `cuda-memcheck`, it should report an "Invalid read of size X" error indicating the out-of-bounds read.
5. Additional Notes:
   - We intentionally avoid initializing the array to emphasize that the out-of-bounds read may produce undefined data.
   - The kernel is kept minimal to isolate the out-of-bounds behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void oob_read_kernel(const int *d_array, int *d_result, int N)
{
    // Only thread 0 performs the out-of-bounds read for simplicity
    if (threadIdx.x == 0)
    {
        // OOB read: access index N, which is one past the last valid element (0-based)
        int val = d_array[N];
        // Store the read value into the result location
        *d_result = val;
    }
}

int main(void)
{
    const int N = 10;
    const int ARRAY_BYTES = N * sizeof(int);

    int *d_array = NULL;
    int *d_result = NULL;
    int h_result = 0;

    // Allocate device memory for the array
    cudaError_t err = cudaMalloc((void**)&d_array, ARRAY_BYTES);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_array failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Allocate device memory for the result
    err = cudaMalloc((void**)&d_result, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_result failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Launch the kernel with 1 block and 1 thread
    oob_read_kernel<<<1, 1>>>(d_array, d_result, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        cudaFree(d_result);
        return EXIT_FAILURE;
    }

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        cudaFree(d_result);
        return EXIT_FAILURE;
    }

    // Copy the result back to host
    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        cudaFree(d_result);
        return EXIT_FAILURE;
    }

    printf("Out-of-bounds read value (index %d): %d\n", N, h_result);

    // Clean up
    cudaFree(d_array);
    cudaFree(d_result);

    return EXIT_SUCCESS;
}
```