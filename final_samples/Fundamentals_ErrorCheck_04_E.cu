/*
Trigger the error by passing a `nullptr` for the event in `cudaEventCreate`.

Thinking:
The goal of this program is to deliberately provoke a CUDA runtime error by invoking `cudaEventCreate` with a null pointer. According to the CUDA runtime API, `cudaEventCreate` expects a pointer to a `cudaEvent_t` variable where it will store the created event handle. Supplying `nullptr` violates the API contract and should return `cudaErrorInvalidValue`. The program will check the return code, print the error message, and exit with a non-zero status to indicate that the expected failure occurred. No additional functionality is required beyond this error-triggering demonstration.

The program will:
1. Include necessary headers.
2. Call `cudaEventCreate(nullptr)`.
3. Inspect the returned error code.
4. Print the error string if it matches the expectation.
5. Exit with a failure code if the error is not produced.

This simple test confirms that passing a null pointer to `cudaEventCreate` is correctly handled by the CUDA runtime and yields the expected error. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaError_t err = cudaEventCreate(nullptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed as expected: %s\n", cudaGetErrorString(err));
        return 1;
    } else {
        fprintf(stderr, "Unexpected success: cudaEventCreate should have failed.\n");
        return 2;
    }
}
