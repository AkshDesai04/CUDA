```c
/*
Aim of the program:
Use `cudaGetSymbolAddress` to get the device address of the constant variable.

Thinking:
The goal is to demonstrate how to retrieve the device pointer for a constant memory
variable declared on the GPU using the CUDA runtime API function
`cudaGetSymbolAddress`.  
To make the example concrete and useful we:
1. Declare a constant array `__constant__ int constArray[10];` on the device.
2. Copy host data into this constant memory using `cudaMemcpyToSymbol`.
3. Call `cudaGetSymbolAddress` with the symbol name to obtain its device address
   (a pointer to the constant memory space).
4. Verify that the pointer works by launching a simple kernel that reads the
   constant array and writes the values into a device output array.
5. Copy the results back to the host and print them, along with the obtained
   device address, to confirm that the constant memory was accessed correctly.

The program includes basic error checking for CUDA API calls and follows
conventional CUDA C/C++ style.  It can be compiled with `nvcc` and will
run on any CUDA-capable GPU.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Constant memory array on the device */
__constant__ int constArray[10];

/* Simple kernel that reads the constant array and writes to an output array */
__global__ void readConstArrayKernel(int *out)
{
    int idx = threadIdx.x;
    if (idx < 10)
    {
        out[idx] = constArray[idx];
    }
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                    \
    do                                                      \
    {                                                       \
        cudaError_t err = call;                             \
        if (err != cudaSuccess)                             \
        {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

int main(void)
{
    /* Host data to copy into constant memory */
    int hConstData[10];
    for (int i = 0; i < 10; ++i)
        hConstData[i] = i * 10;  // Arbitrary values

    /* Copy data to device constant memory */
    CUDA_CHECK(cudaMemcpyToSymbol(constArray, hConstData, sizeof(hConstData)));

    /* Obtain device address of the constant variable */
    void *devPtr = NULL;
    CUDA_CHECK(cudaGetSymbolAddress(&devPtr, constArray));
    printf("Device address of constArray: %p\n", devPtr);

    /* Allocate device memory for output */
    int *dOut = NULL;
    CUDA_CHECK(cudaMalloc((void **)&dOut, sizeof(hConstData)));

    /* Launch kernel to read constant memory */
    readConstArrayKernel<<<1, 10>>>(dOut);
    CUDA_CHECK(cudaGetLastError());  // Check kernel launch status

    /* Copy results back to host */
    int hOut[10];
    CUDA_CHECK(cudaMemcpy(hOut, dOut, sizeof(hOut), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Values read from constArray by kernel:\n");
    for (int i = 0; i < 10; ++i)
        printf("hOut[%d] = %d\n", i, hOut[i]);

    /* Clean up */
    CUDA_CHECK(cudaFree(dOut));

    return 0;
}
```