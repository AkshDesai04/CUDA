/*
Aim: Explain in a comment that because kernel launches are asynchronous, the CPU can continue working on other tasks while the GPU is busy.

Thinking:
We will create a simple CUDA program that demonstrates the asynchronous nature of kernel launches.
The program will:
1. Allocate an array on the device and initialize it on the host.
2. Launch a kernel that performs a trivial operation (e.g., add 1 to each element) but includes a busy‑wait loop to keep the GPU occupied for a noticeable amount of time.
3. Immediately after launching the kernel, the CPU will execute a separate workload (a large loop that sums integers) to show that the host can continue running while the GPU is still processing.
4. We will place a detailed comment right after the kernel launch to explicitly explain that this launch is asynchronous: the CPU does not block and can do other work.
5. Finally, we synchronize and copy the results back to the host to verify correctness.

This example uses the default stream (stream 0). By default, CUDA launches are asynchronous with respect to the host; they only block if we explicitly call cudaDeviceSynchronize or perform a blocking memory copy. The comment will highlight this behavior and show that the CPU can perform work concurrently with the GPU. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that performs a simple addition and includes a busy loop
__global__ void addOneKernel(int *d_arr, int n, int busy_cycles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate work by spinning
        for (int i = 0; i < busy_cycles; ++i) {
            __syncthreads(); // Force synchronization inside the loop to avoid optimization
        }
        d_arr[idx] += 1;
    }
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int BUSY_CYCLES = 1000000; // Number of busy loop iterations to keep GPU busy

    // Allocate host memory
    int *h_arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel asynchronously (default stream)
    addOneKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_arr, N, BUSY_CYCLES);

    // --------------------------------------------------------------------------------
    // At this point, the kernel launch is asynchronous. The CPU thread does not
    // wait for the GPU to finish executing the kernel. It can immediately proceed
    // to the next line of code. This allows the CPU to perform other useful
    // computations or I/O while the GPU continues its work in the background.
    // In this example, we simulate CPU work by running a large loop that sums
    // integers. Because the kernel is still running, the CPU can keep busy
    // without being blocked, demonstrating the concurrency between host and
    // device execution.
    // --------------------------------------------------------------------------------

    // Simulate CPU work while GPU kernel is running
    long long cpu_sum = 0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += i;
    }
    printf("CPU finished its loop, partial sum = %lld\n", cpu_sum);

    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Copy the results back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_arr[i] != i + 1) {
            errors++;
            if (errors <= 10) {
                printf("Error at index %d: expected %d, got %d\n", i, i + 1, h_arr[i]);
            }
        }
    }
    if (errors == 0) {
        printf("All results are correct.\n");
    } else {
        printf("Total errors: %d\n", errors);
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
