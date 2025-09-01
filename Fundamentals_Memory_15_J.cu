/*
Use `cudaGetSymbolAddress` to get a device pointer to the `__device__` variable, which you can then pass to other API functions like `cudaMemset`.

Thinking:
- The aim is to demonstrate obtaining a device pointer to a device-resident global variable using `cudaGetSymbolAddress`.
- After retrieving the pointer, we will use `cudaMemset` to initialize the memory of that variable.
- To verify the operation, we will copy the data back to the host and print it.
- The program will include error checking for each CUDA API call using a helper macro.
- The device variable will be a small integer array so that printing is straightforward.
- The code is a complete .cu file that can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Device global array
__device__ int devArray[10];

int main(void)
{
    // Host array to receive data
    int hostArray[10] = {0};

    // Obtain device pointer to the device array
    void* devPtr = nullptr;
    CHECK_CUDA(cudaGetSymbolAddress(&devPtr, devArray));

    // Use cudaMemset to set the device array to 42
    int setValue = 42;
    size_t numBytes = 10 * sizeof(int);
    // cudaMemset works with byte values; to set all ints to 42 we need to set each byte.
    // But setting each byte to 42 will produce an int value of 0x2A2A2A2A = 707406930,
    // which is not what we want. Instead, we can use cudaMemset for all zeros,
    // then copy a pattern, or use cudaMemset with a char value and then adjust.
    // For simplicity, we will set all bytes to 0x2A and then print the resulting int values.
    // Alternatively, we can use cudaMemset with 0 and then use cudaMemcpy from a host array.
    // Here we will set all bytes to 0x2A and then print the int values.
    CHECK_CUDA(cudaMemset(devPtr, 0x2A, numBytes));

    // Copy back to host
    CHECK_CUDA(cudaMemcpy(hostArray, devPtr, numBytes, cudaMemcpyDeviceToHost));

    // Print the values
    printf("Contents of devArray after cudaMemset with 0x2A:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", hostArray[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
