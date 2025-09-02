```cpp
/*
Implement `C[i] = (A[i] > threshold) ? 1 : 0` (thresholding).

Thinking:
1. I need a CUDA C++ program that demonstrates the thresholding operation on a
   1‑D array of floats (or ints).  The operation is straightforward: for each
   element of the input array A, if the element is greater than a specified
   threshold value, set the corresponding element of C to 1, otherwise set it
   to 0.

2. The program will:
   - Allocate host memory for the input array A and output array C.
   - Initialize A with some example data (for instance, a range of values).
   - Copy A to the device.
   - Launch a CUDA kernel that performs the thresholding on each element.
   - Copy the result back to the host and print a few entries to verify.
   - Clean up device and host memory.

3. I will include basic error checking for CUDA API calls using a helper
   macro `CUDA_CHECK`.  The kernel will be simple: each thread handles one
   index, so I will compute the global thread index from block and thread
   indices and process it only if it is within bounds.

4. For clarity and portability I will use `int` for the output array C, as
   the values are 0 or 1.  The input array A will be of type `float`.  The
   threshold will also be a float.

5. I will write the entire program in a single .cu file.  The main function
   will:
   - Define the size of the array (e.g., 1024 elements).
   - Allocate and initialize host arrays.
   - Allocate device arrays.
   - Copy data to device.
   - Define block and grid sizes (e.g., 256 threads per block).
   - Launch the kernel.
   - Copy result back.
   - Print the first few elements of A and C to confirm correct thresholding.
   - Free device memory and exit.

6. No external libraries (e.g., Thrust) are required; everything will be
   written using raw CUDA API for educational clarity.

The goal is to produce a minimal, complete, and compilable CUDA program that
demonstrates the thresholding operation.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// CUDA kernel that performs thresholding
__global__ void thresholdKernel(const float* A, int* C, int N, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = (A[idx] > threshold) ? 1 : 0;
    }
}

int main()
{
    // Parameters
    const int N = 1024;            // Number of elements
    const float threshold = 0.5f;  // Threshold value

    // Host allocations
    std::vector<float> h_A(N);
    std::vector<int>   h_C(N, 0);

    // Initialize input array with some values
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; // Random values in [0,1)
    }

    // Device allocations
    float* d_A = nullptr;
    int*   d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(int)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    thresholdKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N, threshold);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print some results for verification
    std::cout << "Index\tA[i]\t\tC[i]\n";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << i << "\t" << h_A[i] << "\t" << h_C[i] << "\n";
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```