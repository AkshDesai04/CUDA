```c
/*
Aim of the program:
Write a program that gets the driver version and prints its major and minor versions separately.
(e.g., Driver version 11.4 -> Major: 11, Minor: 4).

Thinking process:
- Use CUDA driver API function `cudaDriverGetVersion` to query the installed CUDA driver version.
- The returned version is an integer where the major number occupies the thousands place
  and the minor number occupies the tens place (e.g., 11000 corresponds to 11.0).
- To extract major: integer division by 1000.
- To extract minor: remainder after dividing by 1000, then divide by 10.
- Implement basic error handling to report failures of the API call.
- Print the result in the requested format.
- Use only standard headers and CUDA headers, no external dependencies.
*/

#include <stdio.h>
#include <cuda.h>

int main(void)
{
    int driver_version = 0;
    cudaError_t err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA driver version: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;

    printf("Driver version %d.%d -> Major: %d, Minor: %d\n",
           major, minor, major, minor);

    return 0;
}
```