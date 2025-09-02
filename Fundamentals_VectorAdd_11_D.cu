```cpp
/*
 * Aim of the program:
 * Is there a performance difference between a standard launch (one thread per element) and a grid-stride launch (many elements per thread) for a problem that fits in one launch? Time both.
 *
 * Thinking:
 * 1. I need to set up a simple kernel that operates on each element of an array. 
 *    The operation should be trivial yet sufficient to produce measurable time, e.g., multiply each element by 2.0f.
 *
 * 2. I will implement two versions of the kernel:
 *    a) Standard launch: Each thread processes exactly one element.
 *    b) Grid-stride launch: Each thread processes multiple elements by looping over the array with a stride equal to the total number of threads in the grid.
 *
 * 3. The problem size should be small enough to fit in one kernel launch for both approaches. 
 *    I'll use N = 1,048,576 elements (2^20) which is comfortably within the limits of typical GPU grid sizes.
 *
 * 4. For timing I will use CUDA events: cudaEventCreate, cudaEventRecord, cudaEventSynchronize, and cudaEventElapsedTime. 
 *    I will time the kernel execution alone, not the memory copies.
 *
 * 5. After each kernel run, I'll copy the data back to host and verify that both kernels produce the same result.
 *
 * 6. I will also add simple error checking after CUDA API calls for robustness.
 *
 * 7. The code will be written in plain C/C++ CUDA style, suitable for compiling with `nvcc`.
 *
 * 8. To keep the example minimal, I will not use any external libraries or advanced features.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 20)          // Number of elements (1,048,576)
#define BLOCK_SIZE 256       // Threads per block

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel 1: Standard launch, one thread per element
__global__ void multiply_one_per_thread(float *d_out, const float *d_in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// Kernel 2: Grid-stride launch, each thread processes multiple elements
__global__ void multiply_grid_stride(float *d_out, const float *d_in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        d_out[i] = d_in[i] * 2.0f;
    }
}

// Host function to verify that two arrays are identical within tolerance
bool verify(const float *a, const float *b, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > eps) {
            fprintf(stderr, "Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    // Allocate host memory
    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_out_std  = (float *)malloc(N * sizeof(float));
    float *h_out_gs  = (float *)malloc(N * sizeof(float));

    if (!h_in || !h_out_std || !h_out_gs) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i) / N;
    }

    // Allocate device memory
    float *d_in, *d_out_std, *d_out_gs;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_std, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_gs,  N * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid size
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------------------
    // Standard launch
    // ----------------------------
    CUDA_CHECK(cudaEventRecord(start));
    multiply_one_per_thread<<<grid, BLOCK_SIZE>>>(d_out_std, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_std_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_std_ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_std, d_out_std, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ----------------------------
    // Grid-stride launch
    // ----------------------------
    CUDA_CHECK(cudaEventRecord(start));
    multiply_grid_stride<<<grid, BLOCK_SIZE>>>(d_out_gs, d_in, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_gs_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_gs_ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gs, d_out_gs, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = verify(h_out_std, h_out_gs, N);
    if (!ok) {
        fprintf(stderr, "Verification failed!\n");
    } else {
        printf("Verification succeeded.\n");
    }

    // Report timings
    printf("Standard launch (1 thread per element): %0.3f ms\n", time_std_ms);
    printf("Grid-stride launch (many elements per thread): %0.3f ms\n", time_gs_ms);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_std));
    CUDA_CHECK(cudaFree(d_out_gs));
    free(h_in);
    free(h_out_std);
    free(h_out_gs);

    return EXIT_SUCCESS;
}
```