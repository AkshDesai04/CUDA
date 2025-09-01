```cuda
/*
Aim of the program: Write a program that gets the driver version and prints its major and minor versions separately. (e.g., Driver version 11.4 -> Major: 11, Minor: 4).

Thinking:
1. The CUDA Runtime API provides cudaDriverGetVersion which returns the driver version as an integer.
2. The returned integer is formatted such that the major version is the thousands digit(s) and the minor version is the next two digits.
   For example, 11.4 is returned as 11040.
3. We can extract the major version by dividing the integer by 1000.
4. The minor version can be extracted by dividing by 10 and then taking modulo 100.
5. We'll check the returned CUDA error status to ensure the function succeeded.
6. Finally, we print the major and minor versions in the required format.

The program is written in C++ with CUDA extensions. It is self-contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int driver_version = 0;
    cudaError_t err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA driver version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Extract major and minor versions
    int major = driver_version / 1000;
    int minor = (driver_version / 10) % 100;

    printf("Driver version: %d.%d\n", major, minor);
    printf("Major: %d\n", major);
    printf("Minor: %d\n", minor);

    return 0;
}
```