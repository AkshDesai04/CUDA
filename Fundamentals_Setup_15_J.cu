/*
Create a C++ class with a destructor that calls `cudaDeviceReset()`. Create an instance of this class in `main`.

I will write a minimal CUDA C++ program that includes the necessary header <cuda_runtime.h> to access the CUDA runtime API. The class, named CudaDeviceResetter, will have an empty constructor and a destructor that calls cudaDeviceReset() to clean up the device at program termination. In main(), I will instantiate this class so that its destructor is invoked automatically when main exits, ensuring the CUDA device is reset. No error checking is added for brevity, but in a real application one might check the return value of cudaDeviceReset().

*/

#include <cuda_runtime.h>
#include <iostream>

class CudaDeviceResetter {
public:
    CudaDeviceResetter() {
        // Constructor can be empty; initialization if needed.
    }
    ~CudaDeviceResetter() {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};

int main() {
    // Create an instance of the class. The destructor will run when it goes out of scope.
    CudaDeviceResetter resetter;

    // Optional: perform any CUDA operations here.

    return 0;
}
