/*
Query and print the total amount of shared memory available per multiprocessor on device 0 in kilobytes (KB).
The program's goal is to fetch the CUDA device properties for device 0 and report the amount of shared memory that is available on each multiprocessor (SM) expressed in kilobytes. 
The plan is to:
1. Include the necessary headers: <cuda_runtime.h> for CUDA API calls and <iostream> for output.
2. In main, call cudaGetDeviceProperties() to fill a cudaDeviceProp structure for device 0.
3. Extract the sharedMemPerMultiprocessor field, which gives the number of bytes of shared memory per SM.
4. Convert this value to kilobytes by dividing by 1024.
5. Print the result to standard output.
6. Handle any CUDA errors by checking the returned cudaError_t and printing an error message if needed.
This simple query demonstrates how to access device-level information via the CUDA runtime API and perform basic unit conversion before reporting the data.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device = 0;
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties for device " << device
                  << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // sharedMemPerMultiprocessor is in bytes; convert to kilobytes
    size_t sharedMemBytes = prop.sharedMemPerMultiprocessor;
    size_t sharedMemKB = sharedMemBytes / 1024;

    std::cout << "Total shared memory per multiprocessor on device " << device
              << ": " << sharedMemKB << " KB" << std::endl;

    return 0;
}
