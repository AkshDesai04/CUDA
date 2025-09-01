/*
 * Aim of the program: Query and print the maximum number of threads per multiprocessor.
 *
 * Thinking process:
 * 1. The program must be written in CUDA C, thus a .cu file that compiles with nvcc.
 * 2. The main requirement is to query the device properties for the maximum number
 *    of threads that can reside on a single multiprocessor.  This value is stored
 *    in the cudaDeviceProp struct as maxThreadsPerMultiProcessor.
 * 3. We need to obtain the number of CUDA capable devices, select one (device 0 for
 *    simplicity), then call cudaGetDeviceProperties to fill a cudaDeviceProp
 *    structure.
 * 4. Once we have the structure, we simply print prop.maxThreadsPerMultiProcessor
 *    using std::cout or printf.  For a small program, std::cout is fine.
 * 5. Add simple error handling: a macro to check CUDA API return codes and abort
 *    with an error message if anything fails.
 * 6. The program should compile with nvcc, so we include <cuda_runtime.h> and
 *    standard headers.  The main function returns 0 on success.
 * 7. Keep the code concise, focusing only on the query and print requirement.
 * 8. Since the output file must be a .cu file, we output the entire file content,
 *    with no additional explanatory text beyond the initial multiline comment.
 */

#include <iostream>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Use device 0
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Maximum threads per multiprocessor on device " << device
              << " (" << prop.name << "): "
              << prop.maxThreadsPerMultiProcessor << std::endl;

    return EXIT_SUCCESS;
}
