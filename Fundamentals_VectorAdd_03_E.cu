```cpp
/*
Aim of the program: Is the boundary check computationally expensive? Why or why not? (It's a very cheap branch, and for full warps, there is often no divergence).

Thinking:
The purpose of this CUDA program is to illustrate how a simple boundary check (if (idx < N)) is performed in a GPU kernel and to provide a way to measure its performance impact. The boundary check is a single conditional branch that only affects a few threads at the edges of a data set. For a fully populated warp (i.e., all 32 threads of a warp execute the same path), the branch does not cause warp divergence; the instruction is executed once for the whole warp and the result is broadcast. When a warp is partially full because the total number of threads is not a multiple of the warp size, only the few inactive threads will evaluate the condition and find it false, but this does not create divergence either because they simply skip the body of the branch. Consequently, the cost of this check is negligible compared to the cost of actual arithmetic or memory operations.

The code below:
1. Allocates an array of integers on both host and device.
2. Copies data to the device.
3. Launches a kernel that writes to each element only if the thread index is within bounds.
4. Uses CUDA events to time the kernel execution, giving an idea of how quickly the kernel runs (including the boundary check).
5. Copies the result back to the host and verifies correctness.

The boundary check is intentionally placed in the kernel, and the program can be compiled and run on any CUDA-capable GPU to observe that the kernel runs quickly, confirming that the branch is inexpensive.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that performs a simple computation only if the thread index is within bounds
__global__ void compute_boundary_check(int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check – cheap branch
    if (idx < N)
    {
        // Example computation: multiply by 2
        d_out[idx] = idx * 2;
    }
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate host memory
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (h_arr == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array (not necessary for this example, but shows typical use)
    for (int i = 0; i < N; ++i)
        h_arr[i] = 0;

    // Allocate device memory
    int *d_arr;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy data from host to device (here data is zeroed, but we do it for completeness)
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, 0);

    // Launch kernel
    compute_boundary_check<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_arr, N);

    // Record stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Simple verification
    bool ok = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_arr[i] != i * 2)
        {
            ok = false;
            printf("Verification failed at index %d: got %d, expected %d\n", i, h_arr[i], i * 2);
            break;
        }
    }
    if (ok)
        printf("Verification succeeded: all values correct.\n");

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
```