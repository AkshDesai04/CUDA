/*
Try to call `cudaFree` on a `nullptr`. Check the return code. Is it an error?

I need to write a CUDA program that simply invokes cudaFree with a null pointer argument and prints the return status. 
The aim is to determine whether cudaFree treats a null pointer as an error or as a no‑op that returns cudaSuccess. 
The standard CUDA documentation says that freeing a null pointer is a no‑op and should return cudaSuccess. 
Therefore, the program should call cudaFree((void*)nullptr), capture the return value, and print it using cudaGetErrorString. 
The main function will return 0 if the call succeeded or exit with non‑zero if an unexpected error occurs. 
I'll include minimal error checking and keep the code simple so that it compiles as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err = cudaFree((void*)nullptr);

    printf("cudaFree((void*)nullptr) returned: %s (%d)\n", cudaGetErrorString(err), err);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaFree returned an error.\n");
        return 1;
    }

    printf("Result: cudaFree on nullptr is NOT an error.\n");
    return 0;
}
