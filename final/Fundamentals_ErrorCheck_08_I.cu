```cuda
/* 
 * Aim of the program (exactly as requested in the prompt):
 * "Time your program with and without `cudaDeviceSynchronize` to see the performance overhead of synchronization."
 *
 * ----- Thinking and design notes -----
 * 1. We want to measure the cost of the host waiting for the GPU to finish a kernel.
 *    The typical way to do this is to launch a kernel and then call cudaDeviceSynchronize().
 *    The difference between the time spent launching the kernel (which is tiny) and the
 *    time spent inside cudaDeviceSynchronize() is the synchronization overhead.
 *
 * 2. To get a realistic kernel execution time, we will also measure the kernel's own
 *    runtime using CUDA events. Events are recorded in the CUDA stream and give a device
 *    timestamp. By recording an event before the launch and one after the launch and
 *    then synchronizing on those events we can compute the time taken by the kernel.
 *
 * 3. To separate the synchronization overhead from the kernel runtime we:
 *    - Launch the kernel.
 *    - Record a CUDA event right before the launch (event_start).
 *    - Record a CUDA event right after the launch (event_stop).
 *    - Synchronize on event_stop and compute elapsed time: kernel_time.
 *
 *    Then:
 *    - Record CPU high‑resolution time (std::chrono::high_resolution_clock::now()) right before
 *      calling cudaDeviceSynchronize().
 *    - Call cudaDeviceSynchronize() to force the host to wait for the GPU.
 *    - Record CPU time again after the call.
 *    - The difference is the sync_overhead.
 *
 * 4. For completeness we also record the total time measured by the host from the
 *    point of launch to the point after cudaDeviceSynchronize(). This total time should
 *    be roughly kernel_time + sync_overhead, though there may be small differences due
 *    to CPU‑side overhead in the kernel launch sequence.
 *
 * 5. The kernel itself does not need to do anything complicated; a simple loop that
 *    performs a few arithmetic operations is sufficient to give it some runtime.
 *
 * 6. Error checking is performed after each CUDA API call to ensure we catch any
 *    unexpected failures. If an error occurs we print a message and exit.
 *
 * 7. The program prints:
 *    - Kernel execution time measured by CUDA events (ms).
 *    - Synchronization overhead measured by CPU timers (us).
 *    - Total time measured by CPU timers from launch to post‑sync (us).
 *    - (Optional) The ratio of sync_overhead to kernel_time to give an idea of relative cost.
 *
 * 8. Build instructions (assuming NVCC is available):
 *      nvcc -O2 -o sync_overhead sync_overhead.cu
 *
 * 9. Run the program; it will output timings. Because the kernel does very little work,
 *    the kernel_time will be very small, and the sync_overhead will dominate. This
 *    demonstrates that waiting for the GPU to finish incurs a measurable cost.
 *
 * 10. The code is fully self‑contained and compiles with a standard CUDA toolkit.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Simple kernel that does a small amount of work to create measurable runtime
__global__ void dummyKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Simple computation to keep the thread busy
        for (int i = 0; i < 1000; ++i)
        {
            data[idx] += i;
        }
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main()
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host and device memory
    int *h_data = (int *)malloc(N * sizeof(int));
    int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(int)));

    // Create CUDA events for timing the kernel execution
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));

    // Record start event, launch kernel, record stop event
    CUDA_CHECK(cudaEventRecord(event_start, 0));
    dummyKernel<<<blocks, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(event_stop, 0));

    // Wait for the stop event to complete (ensures kernel finished)
    CUDA_CHECK(cudaEventSynchronize(event_stop));

    // Calculate kernel execution time (milliseconds)
    float kernelTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, event_start, event_stop));

    // Measure synchronization overhead with CPU timer
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Wait for the GPU to finish all work (including the kernel)
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> syncOverhead = cpu_end - cpu_start;

    // Measure total time from launch to post-sync using CPU timers
    // We will repeat the launch and sync to get a total time
    // (Note: kernelTimeMs is already measured; here we do a separate measurement)
    auto total_start = std::chrono::high_resolution_clock::now();
    // Launch again for total measurement
    CUDA_CHECK(cudaEventRecord(event_start, 0));
    dummyKernel<<<blocks, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(event_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(event_stop));
    auto total_sync_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto total_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> totalElapsed = total_end - total_start;
    std::chrono::duration<double, std::micro> syncAfterLaunch = total_end - total_sync_start;

    // Output results
    printf("Kernel execution time (measured by CUDA events): %f ms\n", kernelTimeMs);
    printf("Synchronization overhead (cudaDeviceSynchronize): %f microseconds\n",
           syncOverhead.count());
    printf("Total elapsed time (launch + sync) measured by CPU: %f microseconds\n",
           totalElapsed.count());
    printf("Sync after launch measured by CPU: %f microseconds\n",
           syncAfterLaunch.count());

    // Optional: ratio
    if (kernelTimeMs > 0.0f)
        printf("Ratio of sync overhead to kernel time: %f\n",
               (float)(syncOverhead.count() / 1000.0f) / kernelTimeMs);
    else
        printf("Kernel time too small to compute ratio.\n");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
```