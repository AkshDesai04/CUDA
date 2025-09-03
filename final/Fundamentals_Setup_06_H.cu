```cuda
/* 
Query and print the surface alignment requirement in bytes.

Thought process:
1. The CUDA runtime provides a device attribute `cudaDevAttrSurfaceAlignment` 
   that specifies the required byte alignment for surfaces on a given device.
2. To query this attribute we use `cudaDeviceGetAttribute`.
3. The workflow:
   - Get the current device with `cudaGetDevice`.
   - Call `cudaDeviceGetAttribute` with the `cudaDevAttrSurfaceAlignment` attribute.
   - Print the returned alignment value.
4. Error handling is added via a simple macro `CUDA_CHECK` to make debugging easier.
5. The program is minimal, portable, and prints the alignment in bytes to standard output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    } while (0)

int main(void) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int alignment;
    CUDA_CHECK(cudaDeviceGetAttribute(&alignment,
                                      cudaDevAttrSurfaceAlignment,
                                      device));

    printf("Surface alignment requirement for device %d: %d bytes\n",
           device, alignment);

    return 0;
}
```