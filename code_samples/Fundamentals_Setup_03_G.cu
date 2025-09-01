/*
Aim: Query and print the maximum dimension of a 1D texture for device 0.

Thoughts and design considerations:
1. We need to use the CUDA Runtime API to query device properties. The relevant struct is cudaDeviceProp, and the field `maxTexture1D` holds the maximum supported size for a 1D texture.
2. Since the prompt specifies device 0, we will set the current device to 0 using `cudaSetDevice(0)`. This ensures any subsequent queries are directed to that device.
3. Error handling is essential: any CUDA API call that can fail should be checked. A small helper macro `CHECK_CUDA_CALL` will simplify this.
4. After obtaining the device properties, we will print:
   - The device name (for clarity).
   - The maximum 1D texture dimension.
   This output will be sent to `stdout` using `printf`.
5. The code is straightforward C with CUDA extensions, so it will compile with `nvcc`. No kernel launches or device memory allocations are needed.
6. The program is self‑contained: it includes only the necessary headers, defines the error checking macro, and contains a single `main` function that performs the query and prints the result.
7. No additional libraries or external dependencies are required.

The final file is a complete `.cu` source that can be compiled with:
    nvcc -o query_texture1d query_texture1d.cu
and run with:
    ./query_texture1d
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API return values
#define CHECK_CUDA_CALL(call)                                              \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

int main(void) {
    // Set the device to device 0
    CHECK_CUDA_CALL(cudaSetDevice(0));

    // Retrieve device properties
    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    // Print the results
    printf("Device 0: %s\n", prop.name);
    printf("Maximum 1D texture dimension: %d\n", prop.maxTexture1D);

    return EXIT_SUCCESS;
}
