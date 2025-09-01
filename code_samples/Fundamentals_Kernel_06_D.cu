```cpp
/* Aim of the program: Use a `float*` instead of an `int*` and write the value `3.14f`.
   Thinking:
   - Allocate a float pointer on the device.
   - Launch a simple kernel that writes the constant 3.14f into the first element.
   - Copy the result back to host memory.
   - Print the value to verify.
   - Include minimal error checking and use a single-thread launch for simplicity. */

#include <iostream>
#include <cuda_runtime.h>

// Simple kernel that writes 3.14f into the first element of the array
__global__ void write_value(float *d_arr)
{
    // Only thread 0 writes
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_arr[0] = 3.14f;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__\
                      << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main()
{
    float *d_arr = nullptr;
    float h_val = 0.0f;

    // Allocate device memory for a single float
    CUDA_CHECK(cudaMalloc(&d_arr, sizeof(float)));

    // Launch kernel with one block of one thread
    write_value<<<1, 1>>>(d_arr);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_val, d_arr, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the value
    std::cout << "Value written by kernel: " << h_val << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```