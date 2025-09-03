/*
Aim: Trigger the error by passing a `nullptr` for the event in `cudaEventCreate`.

Thinking: To trigger a CUDA runtime error, we intentionally pass a null pointer to cudaEventCreate, which expects a pointer to a cudaEvent_t. Passing nullptr violates the API contract, leading to cudaErrorInvalidValue. The program will call cudaEventCreate with a null pointer, check the returned error code, and print it using cudaGetErrorString. This demonstrates error handling.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    // Intentionally pass a nullptr to cudaEventCreate
    cudaEvent_t *event = nullptr;
    cudaError_t err = cudaEventCreate(event);

    // Check and report the error
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        // This branch should not be executed because the call should fail
        cudaEventDestroy(*event);
    }

    // Reset the device before exiting
    cudaDeviceReset();

    return 0;
}
