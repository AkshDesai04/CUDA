```c
/*
Use `cudaEvent`s to time only the kernel execution time.

The purpose of this program is to demonstrate how to measure the execution time of a CUDA kernel using cudaEvent objects, which provide precise timing of GPU operations.  
To achieve this we:

1. Define a simple vector addition kernel `vectorAdd` that runs on the GPU.  
2. Allocate host and device memory, initialize the input arrays on the host, and copy them to the device.  
3. Create two cudaEvent objects: `start` and `stop`.  
4. Record the `start` event just before launching the kernel and the `stop` event immediately after the kernel launch.  
   Because events are recorded on the command queue, this ensures we measure **only** the kernel execution time, excluding data transfer or any other host-side overhead.  
5. Synchronize on the `stop` event, then query the elapsed time between `start` and `stop` using `cudaEventElapsedTime`.  
6. Print the measured time in milliseconds.  
7. Clean up all allocated resources.  

The code also includes basic error checking after CUDA API calls to help identify any failures in memory allocation, kernel launch, or event handling.  
It is written in plain C with CUDA extensions and is fully compilable with `nvcc`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    const int N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc((void **)&d_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc((void **)&d_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc((void **)&d_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy inputs to device
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy to d_A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy to d_B failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Compute elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host (optional)
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy to h_C failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Verify result for a few elements (optional)
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```