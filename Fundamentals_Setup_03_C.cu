/*
Aim: Write a program that queries the available global memory and prints a warning if it is less than a certain threshold (e.g., 4096 MB).

[Thinking]
- The requirement is to create a CUDA C program that uses the CUDA Runtime API to determine how much global memory is free on the device.
- `cudaMemGetInfo` provides two values: the amount of free device memory and the total device memory. These are returned in bytes, so we need to convert them to megabytes for human‑readable output.
- We should handle the possibility that `cudaMemGetInfo` might fail, printing an error message using `cudaGetErrorString`.
- The threshold is 4096 MB (4 GB). If the free memory is less than this, we print a warning message.
- We’ll use standard I/O (`printf`, `fprintf`) to display the information. The program is straightforward and does not require any kernel launches or device selection; it uses the default device.
- Error handling: If memory query fails, exit with a non‑zero status; otherwise, return 0.
- The program is written in CUDA C, so the file extension is `.cu` and it should be compiled with `nvcc`.
- For clarity, we’ll cast the byte values to `unsigned long long` before division to avoid integer truncation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error querying memory info: %s\n", cudaGetErrorString(err));
        return 1;
    }

    unsigned long long free_MB   = free_bytes   / (1024ULL * 1024ULL);
    unsigned long long total_MB  = total_bytes  / (1024ULL * 1024ULL);
    const unsigned long long threshold_MB = 4096ULL; // 4 GB

    printf("Free global memory: %llu MB\n", free_MB);
    printf("Total global memory: %llu MB\n", total_MB);

    if (free_MB < threshold_MB) {
        fprintf(stderr,
                "WARNING: Available global memory (%llu MB) is less than the threshold (%llu MB).\n",
                free_MB, threshold_MB);
    }

    return 0;
}
