/*
The aim of this program is to initialize vector A with the values `i` and vector B with `100-i`. Verify that `C[i]` is always 100.

Thinking:
1. Decide on a vector size N (e.g., 256 or 1024). Since B[i] = 100 - i, values beyond 100 will be negative, but A[i] + B[i] will still equal 100 for all i.
2. Allocate host arrays A, B, C of size N. Initialize A[i] = i, B[i] = 100 - i.
3. Allocate device memory for d_A, d_B, d_C.
4. Copy A and B to device.
5. Launch a CUDA kernel that adds the corresponding elements: C[i] = A[i] + B[i].
6. Copy result back to host C.
7. Verify each element of C equals 100; if any mismatch, report error.
8. Print a success message if all checks pass.
9. Free device and host memory.
10. Include minimal error checking for CUDA calls.
*/

#include <iostream>
#include <cuda_runtime.h>

#define N 256
#define BLOCK_SIZE 128

// CUDA kernel to add two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "    \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main()
{
    // Host allocations
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(100 - i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool allCorrect = true;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_C[i] - 100.0f) > 1e-5f)
        {
            std::cerr << "Mismatch at index " << i << ": "
                      << h_C[i] << " != 100\n";
            allCorrect = false;
            break;
        }
    }

    if (allCorrect)
    {
        std::cout << "All elements verified to be 100. Test passed.\n";
    }
    else
    {
        std::cerr << "Test failed.\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
