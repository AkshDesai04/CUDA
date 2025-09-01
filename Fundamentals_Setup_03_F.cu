/*
Query and print the texture alignment requirement for device 0.

Thinking:
- The CUDA Runtime API provides a function `cudaGetDeviceProperties` that fills a
  `cudaDeviceProp` structure with many pieces of information about a GPU.
- The field `textureAlignment` in that structure gives the alignment requirement
  for textures on that device, expressed in bytes.
- We want to query device index 0, so we pass 0 to `cudaGetDeviceProperties`.
- We must check for errors from the CUDA API call and report them with
  `cudaGetErrorString`.
- For simplicity, we use `printf` to output the alignment value.
- The program includes the necessary headers: `<cuda_runtime.h>` for the CUDA
  runtime API and `<stdio.h>` for standard I/O.
- The `main` function returns 0 on success, 1 on failure.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    printf("Texture alignment requirement for device %d: %zu bytes\n",
           device, prop.textureAlignment);
    return 0;
}
