/*
Aim: Query and print the multiprocessor count for the default device.

Thinking:
The program is a simple CUDA host application that queries the GPU device
properties for the default (device 0) and prints the number of multiprocessors
(also known as Streaming Multiprocessors, SMs).  The key CUDA API calls used
are:

- `cudaGetDeviceCount()` – to make sure at least one device is present.
- `cudaGetDeviceProperties()` – to obtain a `cudaDeviceProp` structure
  containing the `multiProcessorCount` field.
- Standard error handling for CUDA calls via `cudaGetErrorString()`.

The program prints the result to standard output and exits with status 0 on
success or 1 on failure.  No device memory allocation or kernel launch is
required, keeping the code minimal and focused on the query operation.

The program is self‑contained and can be compiled with `nvcc`:

    nvcc -o query_multiprocessor query_multiprocessor.cu

Then executed:

    ./query_multiprocessor

It will output something like:

    Multiprocessor count for device 0: 20
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    int device_id = 0; // default device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return 1;
    }

    printf("Multiprocessor count for device %d: %d\n", device_id,
           prop.multiProcessorCount);

    return 0;
}
