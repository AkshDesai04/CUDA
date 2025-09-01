```cpp
/*
Write a program that simply checks if there is at least one CUDA-capable device and prints 'CUDA is available!' or 'No CUDA devices found.'

Thinking:
- Use CUDA runtime API function cudaGetDeviceCount to obtain number of devices.
- If count > 0, print "CUDA is available!".
- Else, print "No CUDA devices found."
- Include <cuda_runtime.h> and <stdio.h>.
- Use a simple main function that returns 0.
- Handle potential errors from cudaGetDeviceCount gracefully (though for this simple check we can ignore detailed error handling).
- Ensure the program compiles as a .cu file.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        // If the call fails, we assume no CUDA devices.
        printf("No CUDA devices found.\n");
    } else {
        if (deviceCount > 0) {
            printf("CUDA is available!\n");
        } else {
            printf("No CUDA devices found.\n");
        }
    }
    return 0;
}
```