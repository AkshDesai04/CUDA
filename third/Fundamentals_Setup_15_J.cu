/*
Create a C++ class with a destructor that calls `cudaDeviceReset()`. Create an instance of this class in `main`.

I decided to write a small program that defines a class named `CudaManager` with a destructor that calls `cudaDeviceReset()` to ensure that CUDA resources are properly cleaned up when the program exits. In `main`, I instantiate an object of this class so that its destructor will automatically be called at program termination. The program includes minimal error checking and demonstrates how to use the CUDA runtime API from a C++ class. The code is written in a single .cu file and can be compiled with `nvcc`. It does not perform any device operations; its sole purpose is to illustrate the resource cleanup pattern using a class destructor.
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple wrapper class to reset CUDA device upon destruction
class CudaManager {
public:
    CudaManager() {
        // Optional: any initialization can be done here
        std::cout << "CudaManager constructed.\n";
    }

    ~CudaManager() {
        // Ensure CUDA device is reset when the object goes out of scope
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "cudaDeviceReset succeeded.\n";
        }
    }
};

int main() {
    // Create an instance of CudaManager to manage CUDA device lifetime
    CudaManager manager;

    // The rest of the program would go here

    // When main returns, manager will be destroyed and cudaDeviceReset() will be called
    return 0;
}
