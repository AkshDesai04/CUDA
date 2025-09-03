/*
Try to call `cudaFree` on a `nullptr`. Check the return code. Is it an error?

My thinking:
- The goal is to test what happens if we call cudaFree with a null pointer.
- According to CUDA API, passing a null pointer to cudaFree should be a no-op and return cudaSuccess.
- We will write a simple main that calls cudaFree((void*)nullptr) and captures the return code.
- We'll print the error code and the string representation via cudaGetErrorString.
- This will demonstrate whether the function treats nullptr as an error or not.
- The program will be compiled as a .cu file and will use the CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Attempt to free a null pointer
    void* ptr = NULL;
    cudaError_t err = cudaFree(ptr);

    // Print the return code
    printf("cudaFree(NULL) returned: %d (%s)\n",
           err,
           cudaGetErrorString(err));

    // Determine if this is considered an error
    if (err == cudaSuccess) {
        printf("Result: No error. cudaFree(NULL) is allowed.\n");
    } else {
        printf("Result: Error encountered.\n");
    }

    return 0;
}
