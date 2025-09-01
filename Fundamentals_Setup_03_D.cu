/*
 * Query and print the memory bus width of the primary GPU in bits.
 *
 * Aim: Query and print the memory bus width of the primary GPU in bits.
 *
 * Thinking:
 *
 * 1. The program must use CUDA Runtime API to query GPU properties.
 * 2. The key property is `memoryBusWidth` in the `cudaDeviceProp` struct.
 * 3. We need to:
 *    - Include <cuda_runtime.h> and <stdio.h>.
 *    - Determine the number of CUDA devices with `cudaGetDeviceCount`.
 *    - If at least one device is present, select the primary device (device 0).
 *    - Retrieve its properties using `cudaGetDeviceProperties`.
 *    - Print the `memoryBusWidth` field, which is expressed in bits.
 * 4. Error handling:
 *    - Check return status of each CUDA API call.
 *    - If no device is available, print an informative message.
 *    - Use a helper function `checkCudaError` to handle errors.
 *
 * 5. Implementation details:
 *    - Define a simple helper function `checkCudaError` that prints
 *      an error message and exits if a CUDA call fails.
 *    - In `main`, first get device count, then set device 0.
 *    - Call `cudaGetDeviceProperties(&prop, 0)`.
 *    - Print the bus width: `prop.memoryBusWidth` which is an int in bits.
 *    - Return 0 on success.
 *
 * 6. Compile the program with: `nvcc -o buswidth buswidth.cu`
 *    Then run: `./buswidth` to see the bus width in bits.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to check CUDA errors
static void checkCudaError(cudaError_t err, const char *msg)
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

    // Get the number of CUDA devices
    err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "cudaGetDeviceCount");

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the primary device (device 0)
    int deviceId = 0;
    err = cudaSetDevice(deviceId);
    checkCudaError(err, "cudaSetDevice");

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    checkCudaError(err, "cudaGetDeviceProperties");

    // Print memory bus width in bits
    printf("Memory Bus Width of device %d: %d bits\n", deviceId, prop.memoryBusWidth);

    return EXIT_SUCCESS;
}
