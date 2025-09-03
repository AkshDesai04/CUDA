```cpp
/*
Aim of the program: Trigger `cudaErrorInvalidValue` by calling `cudaSetDevice` with a negative device number.

Thinking Process:
1. The goal is to deliberately cause an error by using an invalid argument for cudaSetDevice.
2. According to CUDA runtime documentation, passing a negative device number should result in
   cudaErrorInvalidValue.
3. The program will simply call cudaSetDevice(-1) and then check the return value.
4. If an error occurs, it will print the error string using cudaGetErrorString.
5. The program is minimal, includes necessary headers, and uses the C API of CUDA.
6. We ensure to compile the file with nvcc and it will produce the expected error when run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int negativeDevice = -1;
    cudaError_t err = cudaSetDevice(negativeDevice);

    if (err != cudaSuccess)
    {
        printf("cudaSetDevice failed as expected: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Unexpected success: cudaSetDevice returned cudaSuccess.\n");
    }

    return 0;
}
```