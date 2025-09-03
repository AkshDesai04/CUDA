/*
Aim of the program:
In the host, allocate the struct using `cudaMallocManaged` instead of `cudaMalloc`. Initialize on host, launch kernel to modify, synchronize, and then access directly on host to verify.

Thinking:
1. Use a simple struct (MyStruct) with an int and a float to keep the example concise.
2. Allocate a single instance of MyStruct on the device using cudaMallocManaged so that it is accessible from both host and device memory transparently.
3. Initialize the struct fields on the host; because the memory is unified, the device can see the values immediately.
4. Define a kernel that takes a pointer to MyStruct and modifies its fields (simple arithmetic) to demonstrate that the device can write to the unified memory.
5. Launch the kernel with a single thread (or a simple grid) since we only have one struct.
6. After kernel execution, call cudaDeviceSynchronize() to ensure all device writes are finished before the host reads the values.
7. Finally, print the struct fields on the host to verify that the changes made in the kernel are visible.
8. Use error checking macros for CUDA API calls to make debugging easier.
9. The program is written in C style (no C++ constructs) to adhere to the "CUDA with C" requirement.
10. The file contains only the .cu code, with a descriptive multiline comment at the beginning as requested.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

/* Simple struct with two fields */
typedef struct {
    int   a;
    float b;
} MyStruct;

/* Kernel that modifies the struct */
__global__ void modifyStruct(MyStruct *s)
{
    /* Since there's only one struct, single thread can operate on it */
    s->a += 5;               /* Increment integer */
    s->b *= 2.0f;            /* Double the float */
}

int main(void)
{
    MyStruct *devStruct = NULL;

    /* Allocate unified memory for one MyStruct instance */
    CUDA_CHECK(cudaMallocManaged(&devStruct, sizeof(MyStruct)));

    /* Initialize on host */
    devStruct->a = 10;
    devStruct->b = 3.14f;

    printf("Before kernel launch: a = %d, b = %f\n", devStruct->a, devStruct->b);

    /* Launch kernel to modify the struct */
    modifyStruct<<<1, 1>>>(devStruct);
    CUDA_CHECK(cudaGetLastError());

    /* Wait for device to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Verify results on host */
    printf("After kernel execution: a = %d, b = %f\n", devStruct->a, devStruct->b);

    /* Clean up */
    CUDA_CHECK(cudaFree(devStruct));

    return 0;
}
