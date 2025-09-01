/*
Use the timer in a loop to average the kernel execution time over many runs.

I approached the task by first clarifying the goal: we need a CUDA C program that measures the
execution time of a kernel multiple times, accumulates the times, and then computes an average.
The measurement should use CUDA's event API (`cudaEventRecord` and `cudaEventElapsedTime`) to
obtain accurate timing.  
To keep the example self‑contained I implemented a simple vector operation kernel
(`vectorAdd`) that adds two input arrays element‑wise into an output array.  
The host code:
  * allocates host and device memory,
  * copies input arrays to the device once (outside the timing loop),
  * creates two CUDA events (`start` and `stop`) once,
  * runs the kernel repeatedly in a loop (`N_RUNS` times), recording start and stop,
    synchronizing after each launch, and collecting the elapsed time,
  * calculates and prints the average kernel time,
  * copies the result back for a basic validation step,
  * cleans up resources.

I also added a small helper macro (`checkCudaErrors`) for concise error checking.
The program should compile with `nvcc` and run on any CUDA‑capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )
void check(cudaError_t err, const char *msg, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", msg, file, line);
        fprintf(stderr, "Error string: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const int N_RUNS = 100; // Number of kernel executions to average over
    const size_t bytes = N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_B, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device (outside timing loop)
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Determine execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float totalTime = 0.0f;

    for (int i = 0; i < N_RUNS; ++i) {
        checkCudaErrors(cudaEventRecord(start, 0));
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        checkCudaErrors(cudaEventRecord(stop, 0));
        // Wait for the kernel to finish
        checkCudaErrors(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }

    float avgTime = totalTime / N_RUNS;
    printf("Average kernel execution time over %d runs: %f ms\n", N_RUNS, avgTime);

    // Copy result back to host for verification (optional)
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Simple verification (optional)
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = 0;
            printf("Mismatch at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }
    if (correct) {
        printf("Result verification PASSED.\n");
    } else {
        printf("Result verification FAILED.\n");
    }

    // Clean up
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
