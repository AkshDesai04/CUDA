```cuda
/*
 * Query and print the PCI domain ID for the device.
 *
 * Thinking:
 * 1. The aim is to retrieve the PCI domain ID for each CUDA device present
 *    on the system and print it.  CUDA exposes device attributes via
 *    cudaGetDeviceProperties, which includes the pciDomainID field.
 * 2. We need to query the number of devices with cudaGetDeviceCount.
 * 3. For each device index, call cudaGetDeviceProperties and check for
 *    errors.  The returned cudaDeviceProp struct contains the pciDomainID
 *    (int) field that identifies the PCI domain on which the GPU resides.
 * 4. Print the device number, its name, and the domain ID in a human
 *    readable format.
 * 5. Provide simple error checking for CUDA API calls and exit on failure.
 * 6. This is a simple console application; no GPU kernels are launched.
 * 7. Compile with: nvcc -o pci_domain_query pci_domain_query.cu
 *    Run: ./pci_domain_query
 */
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  PCI domain ID: %d\n", prop.pciDomainID);
    }

    return 0;
}
```