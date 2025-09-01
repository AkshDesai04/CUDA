```c
/* 
 * Aim: Write a program to check and print the L2 cache size of the primary GPU in kilobytes (KB).
 *
 * My thinking:
 * 1. Use the CUDA Runtime API to query the device properties. The relevant field in 
 *    `cudaDeviceProp` is `l2CacheSize`, which holds the L2 cache size in bytes.
 * 2. The "primary GPU" in CUDA parlance refers to the current device selected by the 
 *    runtime. We can obtain its device ID using `cudaGetDevice()`. If that call fails, 
 *    we fall back to device 0 as a reasonable default.
 * 3. After retrieving the properties, we convert the L2 cache size from bytes to kilobytes
 *    by dividing by 1024. This is a simple integer conversion, but we keep it in a
 *    `size_t` to avoid any sign/overflow issues on 64‑bit systems.
 * 4. Print the result using `printf`. If the `l2CacheSize` field reports 0, we print a
 *    message indicating that the cache size is unknown or not reported by the device.
 * 5. Add basic error handling: check the return value of each CUDA API call and exit
 *    with an error message if something goes wrong.
 *
 * The program is written as a single .cu file and can be compiled with `nvcc`:
 *    nvcc -o l2_cache_size l2_cache_size.cu
 * Running it will output something like:
 *    L2 cache size of device 0: 8192 KB
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting current device: %s\n", cudaGetErrorString(err));
        /* fall back to device 0 */
        dev = 0;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties for device %d: %s\n", dev, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (prop.l2CacheSize == 0) {
        printf("L2 cache size for device %d is not reported or is 0 bytes.\n", dev);
    } else {
        size_t l2_kb = prop.l2CacheSize / 1024;
        printf("L2 cache size of device %d: %zu KB\n", dev, l2_kb);
    }

    return EXIT_SUCCESS;
}
```