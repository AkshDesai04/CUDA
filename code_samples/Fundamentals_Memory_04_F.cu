```cpp
/*
Aim: Write a function that takes a device pointer and a size and calls `cudaMemset` to zero it out.

Thinking:
The task is straightforward: create a CUDA host function that takes a device pointer (`void*`) and a size (`size_t`), then invokes `cudaMemset` to set all bytes in that memory region to zero. 
Key points to consider:

1. **Device pointer type** – `cudaMemset` accepts a `void*` for the device pointer, so the function should also use `void*` to keep it generic.
2. **Size unit** – `cudaMemset` expects the number of bytes to set. The size parameter will be provided in bytes, which is typical for CUDA memory operations.
3. **Error handling** – CUDA runtime functions return an error code (`cudaError_t`). It is good practice to check the result of `cudaMemset` and handle errors appropriately. For this example, we’ll use a simple helper macro `CHECK_CUDA` that aborts the program if an error occurs, printing the error message.
4. **Zeroing memory** – `cudaMemset` takes the value to set as an `int`. Passing `0` ensures all bytes are zeroed. No further steps are needed.
5. **Testing** – To demonstrate the function, a small `main` will allocate a device array, call the zeroing function, copy the data back to the host, and print the contents. This verifies that the memory was indeed set to zero.
6. **CUDA version** – The code uses only standard CUDA runtime APIs and should compile with any modern CUDA toolkit.

Implementation details:
- Include `<cuda_runtime.h>` for CUDA API declarations and `<iostream>` for console output.
- Define `CHECK_CUDA` macro that checks a CUDA call’s return value.
- Implement `zeroDeviceMemory(void* d_ptr, size_t size)` which simply calls `cudaMemset(d_ptr, 0, size)` and checks the result.
- In `main`, allocate an integer array on the device, deliberately leave it uninitialized, then call `zeroDeviceMemory`. Afterwards, copy back to host memory and print the values to confirm they are all zero.

No external files or libraries are required beyond the CUDA toolkit. The code is self‑contained and ready to compile with `nvcc`. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error in " << __FILE__ << ":"        \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

// Function to zero out a device memory region.
void zeroDeviceMemory(void* d_ptr, size_t size) {
    CHECK_CUDA(cudaMemset(d_ptr, 0, size));
}

int main() {
    const int N = 10;
    int* d_array = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_array, N * sizeof(int)));

    // Intentionally leave the device memory uninitialized
    // Zero it out using our function
    zeroDeviceMemory(d_array, N * sizeof(int));

    // Copy back to host to verify
    int h_array[N];
    CHECK_CUDA(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    std.out << "Device memory after zeroing: ";
    for (int i = 0; i < N; ++i) {
        std.out << h_array[i] << " ";
    }
    std.out << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_array));
    return 0;
}
```