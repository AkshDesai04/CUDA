/*
Iterate through all devices and print the clock rate for each one.

Thinking:
- The program should use the CUDA Runtime API to query the number of available
  CUDA-capable devices and then retrieve detailed properties for each device.
- The header <cuda_runtime.h> provides the functions cudaGetDeviceCount() and
  cudaGetDeviceProperties(), as well as the cudaDeviceProp structure that
  contains a field called clockRate.  clockRate is expressed in kilohertz.
- For clarity the program will print the device index, its name, and its
  clock rate.  The clock rate can be printed directly in kHz, or converted to
  MHz by dividing by 1000.  In this implementation we simply print the kHz
  value along with a unit label.
- Basic error handling will be added: if cudaGetDeviceCount fails, the
  program prints an error and exits.  If cudaGetDeviceProperties fails for a
  particular device, the program reports the error and continues with the
  next device.
- The main routine is written in C++ using <iostream> for convenience, but it
  could be easily converted to plain C with printf.  The code compiles with
  nvcc and requires no additional libraries beyond the CUDA Runtime.
- No kernels or device code are needed; all work is done on the host.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error: cudaGetDeviceProperties failed for device "
                      << dev << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate << " kHz" << std::endl;
    }

    return EXIT_SUCCESS;
}
