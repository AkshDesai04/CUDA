```cpp
/*
 * Query and print the maximum number of registers available per block.
 *
 * Thinking:
 * 1. The CUDA Runtime API provides cudaDeviceGetAttribute which can retrieve various
 *    device properties. One of the attributes is cudaDevAttrMaxRegistersPerBlock,
 *    which gives the maximum number of 32‑bit registers that a single block can
 *    use on a particular device.
 * 2. We will query this attribute for the first device (device 0). In a multi‑GPU
 *    system, the user could extend this to iterate over all devices.
 * 3. The retrieved value is an integer, so we simply print it to std::cout.
 * 4. Basic error handling: if cudaDeviceGetAttribute fails, print the error
 *    string and exit with a non‑zero status.
 * 5. The program is a minimal CUDA C++ host program; it does not launch any
 *    kernels, so no device code is required. The file extension is .cu because
 *    CUDA compilers accept both C++ and CUDA code in the same file, and it
 *    follows the developer's instruction to produce a .cu file.
 *
 * The program can be compiled with:
 *   nvcc -o query_regs query_regs.cu
 * and run with:
 *   ./query_regs
 *
 * It will output something like:
 *   Maximum number of registers per block on device 0: 65536
 */

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0; // Assuming we query the first device

    // Retrieve the maximum number of registers available per block
    int maxRegsPerBlock = 0;
    cudaError_t err = cudaDeviceGetAttribute(&maxRegsPerBlock,
                                             cudaDevAttrMaxRegistersPerBlock,
                                             device);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device attribute: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Maximum number of registers per block on device "
              << device << ": " << maxRegsPerBlock << std::endl;

    return EXIT_SUCCESS;
}
```