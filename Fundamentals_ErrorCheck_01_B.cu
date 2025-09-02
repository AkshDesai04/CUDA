```c
/*
Use `fprintf(stderr, ...)` to print the error message to standard error instead of standard out.
The purpose of this program is to demonstrate how CUDA error handling can be directed to the
standard error stream using fprintf.  We launch a very simple kernel that prints a message
from each thread.  After the launch we immediately query cudaGetLastError() to see
whether the launch succeeded.  If an error occurs, we write a descriptive message to
stderr via fprintf.  The same approach is applied after cudaDeviceSynchronize() to
capture any errors that happen while the kernel runs.  Finally, if everything
succeeds we print a normal success message to stdout.  This example showcases
proper error handling and ensures that diagnostic output is routed to the
appropriate stream for easier debugging and logging.
*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel()
{
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
    // Launch the kernel
    helloKernel<<<1, 10>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize to catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Kernel finished successfully.\n");
    return 0;
}
```