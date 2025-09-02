/*
Modify your `CHECK` macro to not exit, but instead throw a C++ exception. This allows for more sophisticated error handling in larger applications.

Thinking Process:
1. The goal is to transform the existing `CHECK` macro from exiting the program on a CUDA error to throwing a C++ exception.
2. To do this, the macro will capture the CUDA error code, check if it is not `cudaSuccess`, and if so, build an informative error message that includes the error string, file name, and line number.
3. Since the macro needs to throw an exception, we will use `std::runtime_error`. The message will be constructed with a `std::ostringstream` to concatenate strings and variables cleanly.
4. The macro must be safe to use in both single-line and multi-line contexts, so it will be wrapped in a `do { ... } while(0)` block.
5. The macro will depend on headers `<stdexcept>` and `<sstream>`, which must be included before the macro definition.
6. The rest of the program will provide a small CUDA example that allocates memory, launches a kernel, and demonstrates the use of `CHECK`. Errors will be caught in a `try/catch` block in `main`, printing the exception message instead of terminating abruptly.
7. All code is written in a single `.cu` file as requested, with no additional files or explanations outside the comment block and the actual code.

The final program compiles with `nvcc` and showcases the new exception-throwing `CHECK` macro.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <sstream>

// Macro that checks CUDA errors and throws an exception instead of exiting.
#define CHECK(err) \
  do { \
    cudaError_t err_val = (err); \
    if (err_val != cudaSuccess) { \
      std::ostringstream oss; \
      oss << "CUDA error: " << cudaGetErrorString(err_val) \
          << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(oss.str()); \
    } \
  } while(0)

// Simple kernel that writes indices into an array.
__global__ void dummyKernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = idx;
  }
}

int main() {
  try {
    const int N = 10;
    int *dev_ptr = nullptr;

    // Allocate device memory.
    CHECK(cudaMalloc(&dev_ptr, N * sizeof(int)));

    // Initialize memory to zeros.
    CHECK(cudaMemset(dev_ptr, 0, N * sizeof(int)));

    // Launch kernel.
    const int threadsPerBlock = 4;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocks, threadsPerBlock>>>(dev_ptr, N);

    // Check for kernel launch errors.
    CHECK(cudaGetLastError());

    // Copy results back to host.
    int host_data[N];
    CHECK(cudaMemcpy(host_data, dev_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results.
    std::cout << "Kernel output:" << std::endl;
    for (int i = 0; i < N; ++i) {
      std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up.
    CHECK(cudaFree(dev_ptr));
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
