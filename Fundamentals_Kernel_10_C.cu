/*
Aim: Pass an array of these structs from host to device. Have each thread i access the i-th element of the struct array and print a member.

Thinking: The program will define a simple struct with two members, e.g., an integer id and a float value. The host will allocate an array of these structs, fill it with sample data, and copy it to device memory using cudaMemcpy. A kernel will be launched with one thread per struct element. Inside the kernel, each thread will compute its index (threadIdx.x + blockIdx.x * blockDim.x) and use that index to read the corresponding struct from the device array. Using device printf, the thread will output the desired member. After kernel completion, the host will synchronize and free device memory. The code will include error checking macros to ensure CUDA calls succeed. The example uses device printf, so compilation must be done with -arch=sm_20 or higher.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple struct to be passed to the device */
typedef struct {
    int id;
    float value;
} MyStruct;

/* Macro for error checking */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* Kernel that each thread prints a member of its assigned struct */
__global__ void printStructMember(const MyStruct* d_array, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        /* Print the id member as an example */
        printf("Thread %d: id = %d, value = %f\n", idx, d_array[idx].id, d_array[idx].value);
    }
}

int main(void)
{
    const int numElements = 10;
    const size_t size = numElements * sizeof(MyStruct);

    /* Allocate host array and initialize */
    MyStruct* h_array = (MyStruct*)malloc(size);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < numElements; ++i) {
        h_array[i].id = i;
        h_array[i].value = (float)i * 1.5f;
    }

    /* Allocate device memory */
    MyStruct* d_array;
    CHECK_CUDA(cudaMalloc((void**)&d_array, size));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    /* Determine grid and block dimensions */
    const int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    printStructMember<<<blocks, threadsPerBlock>>>(d_array, numElements);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Wait for device to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Clean up */
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);

    /* Reset device (optional) */
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
