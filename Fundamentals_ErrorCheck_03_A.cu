/*
Explain the difference between `cudaPeekAtLastError` and `cudaGetLastError` in a comment. (Peek does not reset the error state).

The aim of this program is to illustrate how `cudaPeekAtLastError()` and `cudaGetLastError()` behave differently.  
`cudaPeekAtLastError()` simply returns the last CUDA error that occurred, but it does *not* clear the error state.  
`cudaGetLastError()` also returns the last CUDA error, but it *does* reset the error state so that subsequent checks see no error.

In the program we intentionally launch a kernel with an invalid configuration (`<<<0,1>>>`), which generates an error (`cudaErrorInvalidConfiguration`).  
We then:

1. Call `cudaPeekAtLastError()` – this should return the error.
2. Call `cudaGetLastError()` – this should also return the error *and* reset the error state.
3. Call `cudaPeekAtLastError()` again – this should now return `cudaSuccess` because the state has been cleared.

This demonstrates the key difference: peek does not reset, get does.
*/
#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel() {
    // Empty kernel
}

int main() {
    // Intentionally launch with 0 blocks to generate an error
    dummyKernel<<<0, 1>>>();

    // Check error with peek (does not reset)
    cudaError_t errPeek1 = cudaPeekAtLastError();
    printf("After invalid launch, cudaPeekAtLastError() returned: %s\n",
           cudaGetErrorString(errPeek1));

    // Check error with get (resets)
    cudaError_t errGet = cudaGetLastError();
    printf("cudaGetLastError() returned: %s\n", cudaGetErrorString(errGet));

    // Now peek again – should be cudaSuccess because get reset the state
    cudaError_t errPeek2 = cudaPeekAtLastError();
    printf("After cudaGetLastError(), cudaPeekAtLastError() returned: %s\n",
           cudaGetErrorString(errPeek2));

    // Also peek again to show that state remains cleared
    cudaError_t errPeek3 = cudaPeekAtLastError();
    printf("Second peek, still cleared: %s\n", cudaGetErrorString(errPeek3));

    return 0;
}
