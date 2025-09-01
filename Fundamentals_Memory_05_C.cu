/*
Try to call `cudaFree` on a `nullptr`. Check the return code. Is it an error?

Thought process:
- According to the CUDA Runtime API documentation, passing a null pointer to cudaFree is a no-op and should return cudaSuccess.
- The program will call cudaFree(nullptr), capture the return value, and print it along with the corresponding error string.
- We'll use `cudaGetErrorString` to translate the error code to a human-readable message.
- This demonstrates that cudaFree(NULL) is not considered an error and behaves gracefully.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err = cudaFree(nullptr);
    printf("cudaFree(nullptr) returned: %d (%s)\n",
           err,
           cudaGetErrorString(err));
    if (err == cudaSuccess) {
        printf("Result: Not an error.\n");
    } else {
        printf("Result: An error occurred.\n");
    }
    return 0;
}
