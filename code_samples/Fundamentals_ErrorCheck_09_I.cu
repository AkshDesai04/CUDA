```cpp
/*
 * Convert your `checkKernelLaunch` helper function to throw instead of exit.
 *
 * My approach was to first examine the original `checkKernelLaunch` which
 * presumably checks the result of a CUDA kernel launch via `cudaGetLastError()`
 * and then calls `exit(1)` if the error code is not `cudaSuccess`.  The
 * requirement is to change this behavior to throw a C++ exception so that
 * calling code can catch the error and decide how to recover or report it.
 *
 * In CUDA host code it is perfectly fine to throw exceptions; the CUDA
 * runtime API returns error codes and does not interfere with C++ exception
 * handling.  I therefore:
 *
 * 1. Added `#include <stdexcept>` to use `std::runtime_error`.
 * 2. Changed the function signature to `void checkKernelLaunch(const char* file, int line)` so it can report source location.
 * 3. After retrieving the error with `cudaGetLastError()`, if the error is
 *    non‑zero, I construct an informative message containing the file, line,
 *    and the CUDA error string and throw a `std::runtime_error`.
 * 4. Defined a convenience macro `CHECK_KERNEL_LAUNCH()` that expands to
 *    `checkKernelLaunch(__FILE__, __LINE__)` for terse usage after kernel
 *    launches.
 *
 * To demonstrate the helper, I included a trivial vector addition kernel,
 * launched it, and wrapped the launch with `CHECK_KERNEL_LAUNCH()`.  The
 * main function catches any thrown exception and prints the message, then
 * cleans up device memory.  This example shows that the new helper
 * functions exactly like the old one but allows callers to handle errors
 * via exception handling instead of terminating the program immediately.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Helper function that checks the result of the last kernel launch.
// Throws a std::runtime_error instead of exiting the program.
void checkKernelLaunch(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = std::string("CUDA kernel launch error at ") + file + ":" + std::to_string(line)
                          + " - " + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

// Macro to conveniently call checkKernelLaunch with file and line info.
#define CHECK_KERNEL_LAUNCH() checkKernelLaunch(__FILE__, __LINE__)

// Simple kernel that adds two integer arrays.
__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check kernel launch and throw if there's an error
    try {
        CHECK_KERNEL_LAUNCH();
    } catch (const std::runtime_error& e) {
        std::cerr << "Kernel launch failed: " << e.what() << std::endl;
        // Clean up and exit
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        return EXIT_FAILURE;
    }

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Verification failed at index " << i << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "Result verified successfully." << std::endl;

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```