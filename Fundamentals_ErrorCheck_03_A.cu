```cpp
/* 
Aim: Explain the difference between cudaPeekAtLastError and cudaGetLastError.

Thinking:
- The CUDA runtime keeps an error state that can be queried after a kernel launch or any API call.
- cudaPeekAtLastError() returns the last error that occurred, but it does NOT reset the error state.
- cudaGetLastError() also returns the last error that occurred, but in addition it resets the error state to cudaSuccess.
- This distinction is important when you want to inspect errors without clearing them (peek), or when you want to handle an error and clear it for subsequent calls (get).

In this small example we launch a kernel that intentionally causes an error
by using an invalid launch configuration (zero blocks). We then call both
cudaPeekAtLastError and cudaGetLastError to demonstrate the difference in behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Do nothing
}

int main(void)
{
    // Intentionally launch kernel with 0 blocks to cause an error
    dummyKernel<<<0, 1>>>();

    // Check error state with cudaPeekAtLastError (does not reset)
    cudaError_t peekErr = cudaPeekAtLastError();
    if (peekErr != cudaSuccess) {
        printf("cudaPeekAtLastError: %s\n", cudaGetErrorString(peekErr));
    } else {
        printf("cudaPeekAtLastError: no error\n");
    }

    // Check error state with cudaGetLastError (resets error state)
    cudaError_t getErr = cudaGetLastError();
    if (getErr != cudaSuccess) {
        printf("cudaGetLastError: %s\n", cudaGetErrorString(getErr));
    } else {
        printf("cudaGetLastError: no error\n");
    }

    // Call peek again to show that the error state has been reset
    cudaError_t peekErrAfterGet = cudaPeekAtLastError();
    if (peekErrAfterGet != cudaSuccess) {
        printf("cudaPeekAtLastError after get: %s\n", cudaGetErrorString(peekErrAfterGet));
    } else {
        printf("cudaPeekAtLastError after get: no error\n");
    }

    return 0;
}
```