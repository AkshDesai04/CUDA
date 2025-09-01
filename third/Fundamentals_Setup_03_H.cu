```c
/*
Aim of the program:
Query and print the maximum dimensions (width and height) of a 2D texture for device 0.

Thinking process:
- The CUDA runtime API provides cudaGetDeviceProperties which fills a cudaDeviceProp struct
  containing many properties of a device, including the maximum texture 2D width and height
  (fields: maxTexture2DWidth and maxTexture2DHeight).
- We need to query device 0, so first check that at least one CUDA device exists
  using cudaGetDeviceCount.
- If a device exists, retrieve its properties via cudaGetDeviceProperties.
- Extract the maxTexture2DWidth and maxTexture2DHeight from the struct.
- Print them in a human‑readable format using printf.
- Include error checking after each CUDA call so that any failure is reported clearly.
- The program is a plain host program (no device kernels) and can be compiled with
  `nvcc texture_max_dim.cu -o texture_max_dim`.
- The output will be something like: "Maximum 2D texture dimensions (width x height): 65536 x 65536"
  (actual values depend on the GPU).
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                              \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Maximum 2D texture dimensions (width x height) for device %d:\n", device);
    printf("%u x %u\n", prop.maxTexture2DWidth, prop.maxTexture2DHeight);

    return EXIT_SUCCESS;
}
```