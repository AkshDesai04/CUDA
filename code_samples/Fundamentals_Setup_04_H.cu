```c
/*
Aim: Query and print the GPU's core clock rate in GHz.

Thought process:

1. To obtain the core clock rate we use the CUDA Runtime API function `cudaGetDeviceProperties()`.  
   The returned `cudaDeviceProp` structure contains a field `clockRate` which represents the
   clock frequency of the GPU in kilohertz (kHz).

2. Since the requirement is to display the clock rate in gigahertz (GHz), we need to convert
   kHz to GHz.  
   1 kHz = 1,000 Hz, 1 GHz = 1,000,000,000 Hz.  
   Therefore: `GHz = clockRate (kHz) * 1000 / 1,000,000,000 = clockRate / 1,000,000`.

3. The program will:
   - Determine how many CUDA-capable devices are present (`cudaGetDeviceCount`).
   - For each device, retrieve its properties.
   - Convert `clockRate` to GHz and print it with a descriptive message.
   - Handle errors from CUDA API calls gracefully, printing an error message and exiting
     if a call fails.

4. The program is a simple host-only C program; no device kernels are launched.
   It is intended to be compiled with `nvcc` and run on a system with CUDA installed.

5. Edge cases considered:
   - No CUDA devices available → print an informative message and exit.
   - Error in any CUDA call → print the error string returned by `cudaGetErrorString`.

6. The output is a single line per device, e.g.:
   ```
   Device 0: 1.50 GHz
   ```

Now the complete CUDA C source file follows.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue; // proceed to next device
        }

        // prop.clockRate is in kHz
        double ghz = prop.clockRate / 1e6; // convert to GHz

        printf("Device %d (%s): %.3f GHz\n", dev, prop.name, ghz);
    }

    return 0;
}
```