/*
Aim of the program: Pass a nullptr to cudaMemset and use error checking to catch the invalid value error.

Thinking process:
1. The requirement is to intentionally trigger a CUDA error by calling cudaMemset with a null pointer.
2. cudaMemset signature: cudaError_t cudaMemset ( void* devPtr, int value, size_t count );
   Passing devPtr = nullptr should result in cudaErrorInvalidValue.
3. We need to catch this error using error checking.
4. Implement a simple error-checking helper macro/function that checks the return value of CUDA API calls
   and prints a descriptive message if an error occurs.
5. In main, call cudaMemset(nullptr, 0, 1) (any non-zero count works) and then check the error.
6. The program will output the error message, demonstrating that the invalid pointer was caught.
7. The code should be a single .cu file, with standard includes and a minimal main function.
8. No additional output or files; just the program code.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Helper function to check CUDA errors */
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    /* Intentionally pass a nullptr to cudaMemset to trigger an error */
    cudaError_t err = cudaMemset(nullptr, 0, 1);
    checkCudaError(err, "cudaMemset with nullptr");

    /* The program should not reach this point if error checking works correctly */
    printf("cudaMemset succeeded unexpectedly.\n");
    return EXIT_SUCCESS;
}
