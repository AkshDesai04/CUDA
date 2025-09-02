```c
/*
Aim of the program: Use your GpuTimer class to time a vector addition kernel.

Thinking:
1. We need a minimal GpuTimer class that wraps CUDA events to measure elapsed time.
   The class will have cudaEvent_t start and stop, a constructor that creates the events,
   a destructor that destroys them, begin() to record start, end() to record stop,
   and elapsedTime() to compute the elapsed milliseconds using cudaEventElapsedTime.

2. The program will allocate two input vectors A and B and an output vector C on the host.
   Their size (N) will be a moderate value (e.g., 1<<20 elements) to give measurable time.

3. We will copy A and B to device memory, launch a vector addition kernel:
      C[i] = A[i] + B[i]
   The kernel will be launched with enough blocks and threads to cover all elements.

4. We'll use the GpuTimer to time only the kernel launch, not the memory transfers,
   to demonstrate its use for measuring GPU compute time.

5. After kernel execution, copy C back to host, verify results, and print the elapsed
   time in milliseconds.

6. Include error checking macro to simplify CUDA API error handling.

7. The entire program will be a single .cu file with the necessary includes,
   the GpuTimer class definition, kernel definition, and main function.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// GpuTimer class definition
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void begin() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    void end() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }
    float elapsedTime() {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
private:
    cudaEvent_t start, stop;
};

// Vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    GpuTimer timer;
    timer.begin();

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    timer.end();
    float elapsedMs = timer.elapsedTime();
    printf("Vector addition kernel elapsed time: %.3f ms\n", elapsedMs);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = false;
            printf("Mismatch at index %zu: %f vs %f\n", i, h_C[i], expected);
            break;
        }
    }
    if (correct) {
        printf("Results are correct.\n");
    } else {
        printf("Results are incorrect.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```