/*
Aim of the program:
Query and print the texture alignment requirement for device 0.

Thinking process:
1. We need to access CUDA device properties to find the texture alignment requirement.
2. The CUDA Runtime API provides cudaGetDeviceProperties, which fills a cudaDeviceProp struct.
3. The struct contains a member named textureAlignment (size_t) that specifies the minimum texture alignment in bytes for the device.
4. The program will:
   - Set the device to 0 using cudaSetDevice(0).
   - Call cudaGetDeviceProperties(&prop, 0).
   - Print the textureAlignment field.
5. Error handling will be performed by checking return values of CUDA API calls and printing informative messages.
6. The code will be a single .cu file that compiles with nvcc.
7. The program uses <cuda_runtime.h> and <cstdio> for printing.
8. No additional libraries or files are required.
*/

#include <cuda_runtime.h>
#include <cstdio>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            return EXIT_FAILURE;                                   \
        }                                                           \
    } while (0)

int main(void) {
    // Set device 0
    CHECK_CUDA(cudaSetDevice(0));

    // Retrieve device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // Print the texture alignment requirement
    printf("Texture alignment requirement for device 0: %zu bytes\n", prop.textureAlignment);

    return EXIT_SUCCESS;
}
