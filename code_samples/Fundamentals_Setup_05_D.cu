/*
Aim: Query and print the maximum amount of shared memory per block in bytes.

Thought process:
- To obtain the maximum shared memory per block, we need to query the device attribute
  CUDA_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK. This can be retrieved using
  cudaDeviceGetAttribute().
- We will use the default device (device 0). For robustness, we could query the
  current device with cudaGetDevice() and then use that ID.
- The attribute returns the value in bytes as an int. We will print it using
  printf or std::cout.
- Error handling is important: after each CUDA API call, we check the returned
  cudaError_t and print an error message if needed.
- The program is simple: include <cuda_runtime.h> for CUDA runtime API, include
  <cstdio> for printing. The main function will perform the query and print the
  result. Exit with success status.
- Since we are only querying device properties, no kernels or device memory
  allocations are required.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int device = 0;
    // If you want to use the current device, uncomment the following line:
    // cudaGetDevice(&device);

    int maxSharedMem = 0;
    cudaError_t err = cudaDeviceGetAttribute(&maxSharedMem,
                                            cudaDevAttrMaxSharedMemoryPerBlock,
                                            device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum shared memory per block on device %d: %d bytes\n",
           device, maxSharedMem);

    return EXIT_SUCCESS;
}
