/*
Extend the previous program. After copying data to the device, create a second host array of 256 integers. Copy the data from the device array back to this new host array using `cudaMemcpy` with the `cudaMemcpyDeviceToHost` kind. Verify that every element in the new host array matches the original.

Thinking:
1. Allocate a host array `h_orig` of 256 integers and initialize it with some values (e.g., 0 to 255).
2. Allocate device memory `d_arr` for 256 integers.
3. Copy `h_orig` to `d_arr` using `cudaMemcpy` with `cudaMemcpyHostToDevice`.
4. Allocate another host array `h_copy` of 256 integers.
5. Copy data back from `d_arr` to `h_copy` using `cudaMemcpy` with `cudaMemcpyDeviceToHost`.
6. Verify each element of `h_copy` matches the corresponding element in `h_orig`. Print a success message if all match; otherwise, report mismatches.
7. Clean up device memory and exit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const int N = 256;
    int h_orig[N];
    int h_copy[N];
    int *d_arr = NULL;
    size_t size = N * sizeof(int);
    cudaError_t err;

    // Initialize original host array
    for (int i = 0; i < N; ++i)
    {
        h_orig[i] = i;
    }

    // Allocate device memory
    err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_orig, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy data back from device to second host array
    err = cudaMemcpy(h_copy, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (DeviceToHost) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Verify that the copied data matches the original
    int all_match = 1;
    for (int i = 0; i < N; ++i)
    {
        if (h_orig[i] != h_copy[i])
        {
            fprintf(stderr, "Mismatch at index %d: original=%d, copy=%d\n",
                    i, h_orig[i], h_copy[i]);
            all_match = 0;
            break;
        }
    }

    if (all_match)
    {
        printf("All elements match. Data transfer verified successfully.\n");
    }
    else
    {
        printf("Data verification failed.\n");
    }

    // Clean up
    cudaFree(d_arr);

    return 0;
}
