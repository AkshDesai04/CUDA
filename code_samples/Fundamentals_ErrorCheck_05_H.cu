/*
Aim: Write a robust function `bool set_device(int id)` that checks the device count first and only calls `cudaSetDevice` if the ID is valid, returning `false` otherwise.

Thinking:
The function must first query the total number of CUDA-capable devices on the system. This is done with `cudaGetDeviceCount`. If this call fails, we should treat it as an error condition and return `false`, optionally printing an error message.

Once we have the device count, we validate the supplied `id`. A valid ID is a non‑negative integer less than the device count. If the ID is out of range, the function should not attempt to set the device and should return `false`.

If the ID is valid, we call `cudaSetDevice(id)` to set the active device. The result of this call must be checked as well; if it fails, we return `false`. Only when all checks pass do we return `true`.

The implementation will include necessary headers (`cuda_runtime.h`, `stdio.h`, `stdbool.h`) and will use `cudaGetErrorString` to convert CUDA error codes into human‑readable messages. The function returns a plain `bool` to indicate success or failure. This logic ensures the function is safe to call in any context without causing undefined behavior from an invalid device ID or from CUDA runtime failures.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>

/**
 * @brief Sets the current CUDA device to the specified ID after validation.
 *
 * @param id The device ID to set.
 * @return true if the device was successfully set; false otherwise.
 */
bool set_device(int id) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (id < 0 || id >= device_count) {
        fprintf(stderr, "Invalid device ID %d. There are %d device(s) available.\n", id, device_count);
        return false;
    }

    err = cudaSetDevice(id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", id, cudaGetErrorString(err));
        return false;
    }

    return true;
}

/* Example usage (optional)
int main(void) {
    if (set_device(0)) {
        printf("Device 0 set successfully.\n");
    } else {
        printf("Failed to set device 0.\n");
    }
    return 0;
}
*/