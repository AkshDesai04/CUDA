/*
Check if the device supports stream priorities (`streamPrioritiesSupported`).

In CUDA, stream priority support can be queried via the device attribute
`cudaDevAttrStreamPrioritiesSupported`. This attribute returns 1 if the
current device supports stream priorities, otherwise 0. Alternatively, the
`cudaDeviceProp` structure contains a field `streamPrioritiesSupported`,
but using `cudaDeviceGetAttribute` is more direct and works across CUDA
versions.

The program obtains the current device, queries the attribute, checks
for errors, and prints whether stream priorities are supported. This
simple console application demonstrates the check without launching any
kernels or creating streams, making it lightweight and easy to compile
with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device (%d: %s)\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    int support = 0;
    err = cudaDeviceGetAttribute(&support, cudaDevAttrStreamPrioritiesSupported, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to query stream priority support (%d: %s)\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device %d: Stream priorities supported: %s\n",
           device, support ? "Yes" : "No");

    return EXIT_SUCCESS;
}
