/*
Write a program that simply checks if there is at least one CUDA-capable device and prints 'CUDA is available!' or 'No CUDA devices found.'

Thinking:
- The program needs to query the number of CUDA devices present on the system.
- Use the CUDA runtime API function cudaGetDeviceCount to get the device count.
- If the count is greater than zero, output "CUDA is available!".
- If the count is zero, output "No CUDA devices found.".
- The program should be minimal, using standard C headers for printing.
- Include <cuda_runtime.h> to access the CUDA API.
- Handle any potential errors from cudaGetDeviceCount by checking the returned cudaError_t and treating non-zero as failure (although for this simple check we can just ignore detailed error handling and rely on the count).
- This will compile as a .cu file, using nvcc.

Implementation steps:
1. Include headers: stdio.h, cuda_runtime.h.
2. In main, declare int deviceCount.
3. Call cudaGetDeviceCount(&deviceCount).
4. If deviceCount > 0, print success message; else print failure message.
5. Return 0.

Now the code below follows this plan.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        // If we cannot retrieve the device count, we treat it as no device found
        printf("No CUDA devices found.\n");
        return 1;
    }

    if (deviceCount > 0) {
        printf("CUDA is available!\n");
    } else {
        printf("No CUDA devices found.\n");
    }

    return 0;
}
