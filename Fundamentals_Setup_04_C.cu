```cuda
/* 
Aim: Query and print the maximum number of threads per multiprocessor for the default device. 

Thinking:
- Use the CUDA runtime API to query device properties.
- First get the number of devices with cudaGetDeviceCount to ensure at least one device exists.
- Select the default device (device 0) using cudaSetDevice(0). This ensures that subsequent queries refer to the correct device.
- Retrieve the cudaDeviceProp structure for the selected device with cudaGetDeviceProperties.
- The desired value is stored in the field `maxThreadsPerMultiProcessor` of the struct.
- Print this value to stdout in a clear format.
- Include error checking after each CUDA API call. If any call fails, print an error message and exit with a non-zero status.
- The program is simple and uses only standard C headers (`stdio.h`) and the CUDA runtime header (`cuda_runtime.h`).
- The code is self-contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper function to check CUDA errors */
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    int deviceCount = 0;
    cudaError_t err;

    /* Get the number of CUDA-capable devices */
    err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "cudaGetDeviceCount");

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    /* Use the default device (device 0) */
    err = cudaSetDevice(0);
    checkCudaError(err, "cudaSetDevice");

    /* Retrieve properties of the default device */
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, 0);
    checkCudaError(err, "cudaGetDeviceProperties");

    /* Print the maximum number of threads per multiprocessor */
    printf("Maximum threads per multiprocessor for device 0: %d\n",
           deviceProp.maxThreadsPerMultiProcessor);

    return EXIT_SUCCESS;
}
```