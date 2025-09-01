```cpp
/*
Query and print the memory clock rate of the primary CUDA device in kilohertz (kHz).

The goal of this program is to demonstrate how to retrieve and display the
memory clock rate of the main CUDA-capable GPU present on the system.  The
CUDA Runtime API provides a structure `cudaDeviceProp` which contains a
field `memoryClockRate`.  This field is expressed in kilohertz, so the
program can print it directly without any conversion.

Key steps:
1. Include the necessary headers: <cuda_runtime.h> for the CUDA runtime
   API and <cstdio> for standard I/O.
2. In `main()`, select the primary device (device 0) using `cudaGetDeviceProperties`.
3. Handle any potential errors from `cudaGetDeviceProperties` by checking the
   returned `cudaError_t`.
4. If the call is successful, print the value of `memoryClockRate` from the
   `cudaDeviceProp` structure.
5. Exit with status 0 on success, otherwise 1 on failure.

No additional CUDA kernels or device code are required for this query
operation.  The program is self‑contained and can be compiled with nvcc:
   nvcc -o mem_clock_query mem_clock_query.cu
Running the resulting binary will output something like:
   Memory clock rate: 14000kHz
*/
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device_id = 0;  // Primary CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // memoryClockRate is already in kHz
    printf("Memory clock rate of device %d: %dkHz\n", device_id, prop.memoryClockRate);

    return 0;
}
```