/*
 * Aim of the program:
 *   Test with `N = 255`, `N = 256`, and `N = 257` with a block size of 256. Verify correctness for all cases.
 *
 * Thinking process:
 *   - The prompt asks to test with three different array sizes relative to a block size of 256.
 *   - To keep the example simple and focus on CUDA launch configuration, we use a trivial kernel
 *     that increments each element of a vector by one.
 *   - The program will:
 *       1. Allocate device memory for the input vector.
 *       2. Copy a host array initialized to 0..N-1 to device.
 *       3. Launch the kernel with grid size calculated as ceil(N / blockSize).
 *          This tests the edge cases where N < blockSize, N == blockSize, and N > blockSize.
 *       4. Copy the result back to host.
 *       5. Verify that each element is equal to its original value + 1.
 *   - For each of the three N values we perform the above steps and print whether the test passed.
 *   - Simple CUDA error checking macros are used to keep the code readable.
 *   - The code is self‑contained in a single .cu file and uses only standard C++ headers and CUDA runtime APIs.
 */

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << static_cast<int>(err)                  \
                      << " \"" << cudaGetErrorString(err) << "\"\n";        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

const int BLOCK_SIZE = 256;

// Kernel that increments each element by one
__global__ void incrementKernel(int *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

bool runTest(int N)
{
    std::cout << "Running test with N = " << N << " and block size = " << BLOCK_SIZE << std::endl;

    int *h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i; // initialize to 0..N-1
    }

    int *d_data = nullptr;
    size_t size = N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    incrementKernel<<<gridSize, BLOCK_SIZE>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != i + 1) {
            std::cerr << "Verification failed at index " << i
                      << ": expected " << i + 1 << ", got " << h_data[i] << std::endl;
            success = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;

    if (success) {
        std::cout << "Test passed.\n";
    } else {
        std::cout << "Test failed.\n";
    }
    std::cout << std::endl;
    return success;
}

int main()
{
    // Set device (optional if only one device)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    bool allPassed = true;
    allPassed &= runTest(255);
    allPassed &= runTest(256);
    allPassed &= runTest(257);

    if (allPassed) {
        std::cout << "All tests succeeded.\n";
        return EXIT_SUCCESS;
    } else {
        std::cout << "Some tests failed.\n";
        return EXIT_FAILURE;
    }
}
