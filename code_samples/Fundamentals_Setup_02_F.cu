```cpp
/*
Aim of the program:
Write a program that finds the device with the highest compute capability and prints its name and index.

Thinking and design notes:
1. Use CUDA runtime API to query the number of CUDA-capable devices via cudaGetDeviceCount.
2. Iterate over each device index from 0 to count-1.
3. For each device, obtain its properties using cudaGetDeviceProperties, which returns a cudaDeviceProp structure.
4. Compute capability is represented by major and minor numbers (e.g., compute capability 7.5 has major=7, minor=5).
5. Keep track of the device with the highest compute capability. Compare first by major, then by minor if majors are equal.
6. Store the index and name of the best device found.
7. After iterating, output the index and name using std::cout.
8. Handle error cases: if cudaGetDeviceCount returns an error or returns 0 devices, print appropriate messages.
9. Ensure the program compiles with NVCC and runs on any system with CUDA installed.
10. The code will be contained in a single .cu file, with all necessary includes and a main() function.

*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int bestDeviceIdx = -1;
    int bestMajor = -1;
    int bestMinor = -1;
    char bestName[256] = {0};

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << i
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue; // skip this device
        }

        // Update if this device has a higher compute capability
        if (prop.major > bestMajor || (prop.major == bestMajor && prop.minor > bestMinor)) {
            bestMajor = prop.major;
            bestMinor = prop.minor;
            bestDeviceIdx = i;
            std::strncpy(bestName, prop.name, sizeof(bestName)-1);
            bestName[sizeof(bestName)-1] = '\0'; // ensure null-termination
        }
    }

    if (bestDeviceIdx == -1) {
        std::cerr << "Could not determine the best device." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Best device index: " << bestDeviceIdx << std::endl;
    std::cout << "Device name   : " << bestName << std::endl;
    std::cout << "Compute capability: " << bestMajor << "." << bestMinor << std::endl;

    return EXIT_SUCCESS;
}
```