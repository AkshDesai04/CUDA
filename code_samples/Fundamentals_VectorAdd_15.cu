/*
Aim: Implement a 'fused' kernel that performs two operations at once. For example, C[i] = A[i] + B[i] and D[i] = A[i] - B[i]. This can improve performance by increasing arithmetic intensity.

Thinking:
1. We want a single kernel launch that computes both addition and subtraction to reduce overhead.
2. Each thread will read A[i] and B[i] once, perform two arithmetic operations, and write results to C[i] and D[i].
3. This increases arithmetic intensity (operations per memory load) and reduces global memory traffic compared to launching two separate kernels.
4. We'll use float arrays for simplicity but the kernel can be templated if needed.
5. We'll provide basic error checking and timing using cudaEvent_t.
6. Memory allocations: host arrays (h_A, h_B, h_C, h_D) and device arrays (d_A, d_B, d_C, d_D).
7. We'll initialize h_A and h_B with some values (e.g., sequential or random) for demonstration.
8. After kernel execution, we copy back results and verify against CPU-computed reference.
9. We'll include a simple command line argument to specify array size; default to 1<<20 elements.
10. Kernel launch configuration: 256 threads per block; number of blocks computed to cover N elements.
11. We'll use cudaMallocManaged if we want to simplify memory management, but here we use cudaMalloc and cudaMemcpy for clarity.
12. Ensure the code compiles with nvcc and runs on any CUDA-capable device.

The program demonstrates a fused kernel that performs addition and subtraction in a single pass, showcasing how combining multiple operations can improve performance by reducing memory traffic and kernel launch overhead.

Now the actual CUDA C code follows:
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                          \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "."       \
                      << std::endl;                                               \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }

// Fused kernel: compute C[i] = A[i] + B[i] and D[i] = A[i] - B[i]
__global__ void fusedKernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            float* __restrict__ D,
                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

// CPU reference implementation for verification
void referenceAddSub(const float* A, const float* B, float* C, float* D, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
        D[i] = A[i] - B[i];
    }
}

int main(int argc, char* argv[]) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20); // Default 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);
    float *h_D_ref = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D || !h_C_ref || !h_D_ref) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize host arrays with random data
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // CPU reference calculation
    referenceAddSub(h_A, h_B, h_C_ref, h_D_ref, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch fused kernel
    fusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Fused kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    const float eps = 1e-5f;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > eps || fabs(h_D[i] - h_D_ref[i]) > eps) {
            std::cerr << "Mismatch at index " << i
                      << ": GPU (" << h_C[i] << ", " << h_D[i]
                      << ") vs CPU (" << h_C_ref[i] << ", " << h_D_ref[i] << ")\n";
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Verification PASSED." << std::endl;
    } else {
        std::cout << "Verification FAILED." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_C_ref);
    free(h_D_ref);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
