/*
Create a `CHECK_CUDA_THROW` macro for your new exception-based error handling.
The goal is to provide a convenient way to check the return value of CUDA runtime API calls
and throw a C++ exception instead of propagating error codes. This aligns with modern
C++ practices where exceptions are preferred over manual error checks.

Approach:
1. Define a custom exception class `CudaException` derived from `std::runtime_error`.  
2. Create a macro `CHECK_CUDA_THROW(expr)` that evaluates the CUDA API call `expr`,
   checks if the returned `cudaError_t` is not `cudaSuccess`, and throws a `CudaException`
   with the corresponding error string.
3. Provide a simple kernel to demonstrate usage, and a `main` function that:
   - Allocates device memory,
   - Launches the kernel,
   - Copies data back,
   - Frees resources.
   All CUDA API calls in the flow are wrapped with `CHECK_CUDA_THROW`.
4. The code compiles with `nvcc` and uses standard C++ features only.

This design keeps the macro concise, readable, and reusable across the codebase,
and the custom exception class can be extended later if needed.
*/

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

/* Custom exception type for CUDA errors */
class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& msg) : std::runtime_error(msg) {}
};

/* Macro that checks CUDA API return status and throws an exception on failure */
#define CHECK_CUDA_THROW(expr)                                         \
    do {                                                              \
        cudaError_t err__ = (expr);                                   \
        if (err__ != cudaSuccess) {                                  \
            throw CudaException(                                      \
                std::string("CUDA error: ") +                        \
                cudaGetErrorString(err__) + " at " +                 \
                __FILE__ + ":" + std::to_string(__LINE__));           \
        }                                                             \
    } while (0)

/* Simple kernel that writes a constant value to an array */
__global__ void set_values(int* d_arr, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] = val;
    }
}

int main() {
    try {
        const int N = 1 << 20;  // 1M elements
        const int val = 42;

        /* Allocate device memory */
        int* d_arr = nullptr;
        CHECK_CUDA_THROW(cudaMalloc(&d_arr, N * sizeof(int)));

        /* Launch kernel */
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        set_values<<<blocksPerGrid, threadsPerBlock>>>(d_arr, val, N);
        /* Ensure kernel launch succeeded */
        CHECK_CUDA_THROW(cudaGetLastError());

        /* Copy result back to host */
        int* h_arr = new int[N];
        CHECK_CUDA_THROW(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

        /* Verify a few values */
        for (int i = 0; i < 5; ++i) {
            std::cout << "h_arr[" << i << "] = " << h_arr[i] << std::endl;
        }

        /* Clean up */
        delete[] h_arr;
        CHECK_CUDA_THROW(cudaFree(d_arr));

        std::cout << "CUDA operations completed successfully." << std::endl;
    }
    catch (const CudaException& e) {
        std::cerr << "Caught CudaException: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
