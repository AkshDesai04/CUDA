/*
Query and print the total amount of constant memory available on device 0 in kilobytes (KB).
To achieve this, I considered the CUDA runtime API which provides device properties via cudaGetDeviceProperties.
The constant memory size is stored in the struct member constantMemorySize, which is measured in bytes.
I will convert this value to kilobytes by dividing by 1024.0. The program will:

1. Set the device to 0 using cudaSetDevice.
2. Retrieve the device properties into a cudaDeviceProp struct.
3. Extract the constantMemorySize field.
4. Convert it to KB.
5. Print the result using printf.

I also added basic error checking after each CUDA call to ensure robustness. The program is simple and self-contained,
requiring only the CUDA runtime library to compile. It will compile with `nvcc` and run on any system with an appropriate
GPU supporting CUDA. The output will be a single line indicating the constant memory size in kilobytes.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    size_t constMemBytes = prop.totalConstMem; // Constant memory in bytes
    double constMemKB = static_cast<double>(constMemBytes) / 1024.0;

    printf("Device %d constant memory: %.2f KB\n", device, constMemKB);

    return 0;
}
