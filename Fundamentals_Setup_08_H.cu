/*
Aim: Print the total constant memory on the device in bytes.
The goal of this CUDA program is to query the device properties using cudaGetDeviceProperties and retrieve the total amount of constant memory available on the GPU. The constant memory size is stored in the totalConstMem field of the cudaDeviceProp structure. We will print this value in bytes to the standard output.

The reasoning behind this program:
1. We include <cuda_runtime.h> and <stdio.h> for CUDA runtime API and I/O functions.
2. In main, we call cudaGetDeviceProperties for device 0 (the first GPU).
3. We check for any errors from the CUDA call. If an error occurs, we print an error message and exit with failure.
4. If the call succeeds, we retrieve the totalConstMem field which represents the size of the constant memory in bytes.
5. Finally, we print this size using printf and return success.

This program is simple and portable for any CUDA-capable GPU. It demonstrates basic error handling for CUDA runtime API calls and accesses a device property.

The code below follows the above plan.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Total constant memory on device 0: %zu bytes\n", prop.totalConstMem);
    return 0;
}
