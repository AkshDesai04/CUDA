```c
/*
Aim of the program: Define a struct that contains a small, fixed-size array (e.g., float vec[3]). Pass it by value and have the kernel print the array elements.

Thinking:
- We need a simple CUDA program that demonstrates passing a small struct by value to a device kernel.
- The struct will have a fixed-size array of three floats.
- The kernel will receive this struct directly as a parameter (by value) and use device-side printf to output each element.
- Device printf requires compute capability >= 2.0; we’ll assume the target device meets this requirement.
- The host program will create an instance of the struct, initialize it, and launch the kernel with that instance.
- Because we pass the struct by value, CUDA will automatically copy the data from host to device as part of the kernel launch arguments.
- We’ll use standard CUDA runtime API calls, add minimal error checking, and synchronize before exiting so the printf output is flushed.
- No external files or resources are required; the entire program is self-contained in a single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

/* Structure containing a small fixed-size array */
struct Vec3 {
    float vec[3];
};

/* Kernel that receives Vec3 by value and prints its elements */
__global__ void printVecKernel(Vec3 v) {
    /* Device-side printf requires a format string that matches the arguments */
    printf("vec[0] = %f, vec[1] = %f, vec[2] = %f\n", v.vec[0], v.vec[1], v.vec[2]);
}

/* Simple error checking macro */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",       \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    /* Host instance of Vec3 */
    Vec3 h_vec = { { 1.0f, 2.0f, 3.0f } };

    /* Launch kernel with a single thread */
    printVecKernel<<<1, 1>>>(h_vec);

    /* Wait for GPU to finish before accessing on host */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Clean up and exit */
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
```