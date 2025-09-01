```c
/* 
Query and print the memory bus width in bits.

--- Thinking Process ---

1. The goal is to write a CUDA C program that reports the memory bus width (in bits) of the GPU device(s). 
   The memory bus width is available through the CUDA Runtime API in the `cudaDeviceProp` structure as the
   `memoryBusWidth` field.

2. To achieve this we need:
   - Include the CUDA runtime header (`cuda_runtime.h`) and standard C++ headers for output.
   - Call `cudaGetDeviceCount()` to know how many devices are present.
   - For each device, call `cudaGetDeviceProperties()` to fetch the `cudaDeviceProp` struct.
   - Extract the `memoryBusWidth` member and print it in a readable format.

3. Error handling:
   - Wrap CUDA API calls with a helper macro or inline function that checks the return status and aborts with a helpful message if an error occurs.
   - This ensures that if the device query fails, the user receives a clear error.

4. Output format:
   - For each device we will print: "Device X: memory bus width = Y bits".
   - Use `std::cout` for simplicity.

5. The program should compile with `nvcc`. We keep it simple, using a single `main` function.

6. No external dependencies beyond the CUDA runtime and the standard library.

--- End of Thinking ---
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro to check CUDA API return values
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in file '" << __FILE__ << "' at line "  \
                      << __LINE__ << ": " << cudaGetErrorString(err)          \
                      << " (" << err << ")." << std::endl;                    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << "memory bus width = " << prop.memoryBusWidth
                  << " bits" << std::endl;
    }

    return EXIT_SUCCESS;
}
```