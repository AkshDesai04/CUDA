/*
Create a C++ class with a destructor that calls `cudaDeviceReset()`. Create an instance of this class in `main`.

Thought process:
- The program requires a C++ class whose destructor performs the CUDA cleanup by calling `cudaDeviceReset()`.
- To use `cudaDeviceReset()`, we need to include `<cuda_runtime.h>`.
- The destructor should be defined inside the class to ensure it is called when the object goes out of scope (i.e., at the end of `main`).
- The `main` function will simply instantiate the class; the destructor will run automatically.
- We will not add any additional logic beyond the required destructor call to keep the example concise and focused on the requested behavior.
- The file is intended to be compiled with NVCC (CUDA compiler), so it will have a `.cu` extension.
- No error handling is added beyond the basic call; for a robust application you might check the return value, but the prompt doesn't require it.
*/

#include <cuda_runtime.h>
#include <iostream>

class CudaDeviceResetter {
public:
    CudaDeviceResetter() {
        // Constructor can be empty or perform initialization if needed.
        std::cout << "CudaDeviceResetter constructed." << std::endl;
    }

    ~CudaDeviceResetter() {
        // Call cudaDeviceReset() to reset the device before program exit.
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "cudaDeviceReset called successfully." << std::endl;
        }
    }
};

int main() {
    // Create an instance of the class; destructor will be invoked when it goes out of scope.
    CudaDeviceResetter resetter;
    // The program can perform other CUDA operations here if desired.
    return 0;
}
