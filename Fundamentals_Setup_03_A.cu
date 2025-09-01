/*
Aim: Query and print the total amount of constant memory available on device 0 in kilobytes (KB).

Thinking:
1. We need to use the CUDA Runtime API to obtain device properties. The struct `cudaDeviceProp` contains a member `totalConstMem` which gives the size of constant memory in bytes.
2. Since the requirement is to query device 0, we will directly pass device index 0 to `cudaGetDeviceProperties`.
3. After retrieving the size in bytes, convert it to kilobytes by dividing by 1024.0. We use a double for a nicer decimal representation.
4. Print the result using `printf` for simplicity. The output format will be: "Device 0 has X.XX KB of constant memory."
5. Include basic error checking: if `cudaGetDeviceProperties` fails, print an error message to `stderr` and exit with a non-zero status.
6. The program will be a simple `main` function; no additional CUDA kernel launches are required.

This program can be compiled with `nvcc`:
    nvcc -o query_const_mem query_const_mem.cu
and run as:
    ./query_const_mem
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    size_t totalConstMemBytes = prop.totalConstMem;
    double totalConstMemKB = (double)totalConstMemBytes / 1024.0;

    printf("Device %d has %.2f KB of constant memory.\n", device, totalConstMemKB);
    return 0;
}
