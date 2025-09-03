/*
Aim of the program: Create a C++ class with a destructor that calls `cudaDeviceReset()`. Create an instance of this class in `main`.

Thinking:
1. The goal is to ensure that when the program terminates, the CUDA device is properly reset.
2. A C++ class with a destructor is ideal for this because the destructor will automatically be invoked when the object goes out of scope (i.e., at program exit).
3. In the destructor, call `cudaDeviceReset()` and handle any errors by printing them to `stderr`.
4. Include the necessary CUDA header `<cuda_runtime.h>` and standard headers for I/O.
5. In `main`, instantiate the class (e.g., `CudaResetter resetter;`). This instance will live until the end of `main`, triggering the destructor and thus the device reset.
6. The code is minimal but functional; it compiles with `nvcc` and demonstrates the required behavior.
*/

#include <cuda_runtime.h>
#include <iostream>

class CudaResetter {
public:
    CudaResetter() {
        // Optional: initialize something if needed
    }

    ~CudaResetter() {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};

int main() {
    CudaResetter resetter;  // Instance created here

    // Optional: perform CUDA operations here

    return 0;  // Destructor of resetter will be called here, resetting the device
}
