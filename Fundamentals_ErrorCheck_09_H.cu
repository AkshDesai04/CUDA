```cpp
/*
Aim: Create a `CHECK_CUDA_THROW` macro for your new exception-based error handling.

Thinking:
- We want a macro that takes a CUDA runtime API call, checks its return status, and throws an exception if the call failed.
- Use cudaError_t to capture the status.
- If the status is not cudaSuccess, we want to throw an exception with a descriptive message including the error string from cudaGetErrorString, and optionally file and line information for debugging.
- The macro should be safe to use in both single statements and in control flow blocks. A do-while(0) wrapper is standard.
- We need to include <cuda_runtime.h> for CUDA API, and <stdexcept> and <string> for exceptions.
- Optionally define a custom CudaException class derived from std::runtime_error for clearer type, but we can just use std::runtime_error.
- Demonstrate usage by writing a simple vector addition kernel, allocating device memory, copying data, launching the kernel, copying back, and freeing memory. Each CUDA call will be wrapped by CHECK_CUDA_THROW.
- Wrap the main body in a try/catch block to catch the exception and print the message.
- Ensure the program compiles with nvcc and runs on a CUDA-capable device.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Macro to check CUDA runtime API calls and throw an exception on error
#define CHECK_CUDA_THROW(call)                                               \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("CUDA error: ") + cudaGetErrorString(err) +    \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                    \
    } while (0)

// Simple vector addition kernel
__global__ void addKernel(int *c, const int *a, const int *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int N = 256;
    const size_t bytes = N * sizeof(int);

    // Host memory allocation
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Device memory allocation
    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_c = nullptr;
    try {
        CHECK_CUDA_THROW(cudaMalloc((void**)&d_a, bytes));
        CHECK_CUDA_THROW(cudaMalloc((void**)&d_b, bytes));
        CHECK_CUDA_THROW(cudaMalloc((void**)&d_c, bytes));

        // Copy data from host to device
        CHECK_CUDA_THROW(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_THROW(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 128;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);

        // Check for kernel launch errors
        CHECK_CUDA_THROW(cudaGetLastError());

        // Copy result back to host
        CHECK_CUDA_THROW(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

        // Verify results
        for (int i = 0; i < N; ++i) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                std::cerr << "Verification failed at index " << i << std::endl;
                return EXIT_FAILURE;
            }
        }
        std::cout << "Vector addition successful. All results verified." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        // Free device memory in case of exception
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        return EXIT_FAILURE;
    }

    // Clean up device memory
    CHECK_CUDA_THROW(cudaFree(d_a));
    CHECK_CUDA_THROW(cudaFree(d_b));
    CHECK_CUDA_THROW(cudaFree(d_c));

    // Clean up host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return EXIT_SUCCESS;
}
```