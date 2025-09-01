```cuda
/*
  Aim: Query and print the PCI domain ID for the device.

  Thought process:
  1. We need to use CUDA Runtime API to access device properties.
  2. The cudaDeviceProp structure contains pciDomainID and pciBusID fields.
  3. We'll first get the number of CUDA-capable devices via cudaGetDeviceCount.
  4. For each device, we call cudaGetDeviceProperties to obtain the structure.
  5. We'll print the device index, name, PCI bus ID, and PCI domain ID.
  6. Basic error checking will be performed to ensure API calls succeed.
  7. The program will compile with nvcc and run on any system with CUDA installed.
  8. Output will be printed to stdout using printf.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaGetDeviceProperties for device %d failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  PCI Bus ID:  %02X\n", prop.pciBusID);
        printf("  PCI Domain ID: %04X\n", prop.pciDomainID);
        printf("\n");
    }

    return 0;
}
```