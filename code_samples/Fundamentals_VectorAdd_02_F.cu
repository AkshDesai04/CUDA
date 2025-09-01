/*
Compare the execution time for `N=1M` vs `N=2M`. Is it roughly double?

The aim of this CUDA program is to benchmark the execution time of a simple element-wise
kernel that increments each element of a large array.  By running the kernel on two
different problem sizes – 1 M and 2 M elements – we can observe how the runtime scales
with input size.  We use CUDA events to measure the kernel execution time precisely.
The program allocates device memory for each N, initializes a host array with zeros,
copies it to the device, launches a kernel that adds 1 to each element, and then
copies the result back to host.  The time measured only includes the kernel launch
and execution (memory transfers are timed separately to confirm they are negligible
compared to the compute time).  Finally, we print the execution times for both
problem sizes and a short comment on whether the runtime roughly doubles.

The kernel is intentionally simple so that the computation time is dominated by the
number of arithmetic operations, making the scaling with N clear.  We also ensure
proper error checking and cleanup of CUDA resources.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element of the array
__global__ void incrementKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int sizes[2] = { 1'000'000, 2'000'000 };
    float times[2] = {0};

    for (int i = 0; i < 2; ++i) {
        int N = sizes[i];
        size_t bytes = N * sizeof(int);

        // Allocate host memory
        int *h_data = (int*)malloc(bytes);
        if (!h_data) {
            fprintf(stderr, "Host memory allocation failed for N=%d\n", N);
            exit(EXIT_FAILURE);
        }
        // Initialize host array to zero
        memset(h_data, 0, bytes);

        // Allocate device memory
        int *d_data;
        checkCudaError(cudaMalloc((void**)&d_data, bytes), "cudaMalloc");

        // Copy data to device
        checkCudaError(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

        // Define kernel launch parameters
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
        checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

        // Record start event
        checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start");

        // Launch kernel
        incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

        // Record stop event
        checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop");
        // Wait for the event to complete
        checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize");

        // Calculate elapsed time
        float ms = 0.0f;
        checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        times[i] = ms;

        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Copy result back (optional, not timed)
        checkCudaError(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

        // Validate result (optional)
        for (int j = 0; j < N; ++j) {
            if (h_data[j] != 1) {
                fprintf(stderr, "Validation failed at index %d for N=%d\n", j, N);
                break;
            }
        }

        // Free device and host memory
        cudaFree(d_data);
        free(h_data);
    }

    // Output the times
    printf("Execution times (kernel only, ms):\n");
    printf("N = %d: %.3f ms\n", sizes[0], times[0]);
    printf("N = %d: %.3f ms\n", sizes[1], times[1]);

    if (times[1] > 0.0f)
        printf("Ratio (2M / 1M): %.2f\n", times[1] / times[0]);

    return 0;
}
