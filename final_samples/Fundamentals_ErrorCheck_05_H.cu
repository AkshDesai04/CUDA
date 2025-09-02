/*
 * Aim of the program: Write a robust function bool set_device(int id) that checks the device count first and only calls cudaSetDevice if the ID is valid, returning false otherwise.
 *
 * Thinking:
 * 1. The function must query the number of CUDA devices using cudaGetDeviceCount.
 * 2. If the query fails (e.g., driver not available), the function should return false.
 * 3. We need to handle invalid IDs: negative IDs or IDs >= deviceCount.
 * 4. If the ID is valid, call cudaSetDevice(id) and return true only if that call succeeds.
 * 5. All CUDA API calls return a cudaError_t; we check against cudaSuccess.
 * 6. The function should be simple, reusable, and not produce side effects beyond setting the device.
 * 7. Include necessary headers: <cuda_runtime.h> and <cstdio> for potential debugging output.
 * 8. Provide a small main() example that demonstrates usage: it prints the number of devices and attempts to set a device based on user input or a hardcoded value.
 * 9. Ensure the code compiles as a .cu file and links against the CUDA runtime library.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <stdbool.h>

/**
 * @brief Sets the CUDA device to the specified ID if it is valid.
 *
 * This function first queries the number of available CUDA devices.
 * If the query fails, or if the provided ID is out of range
 * (negative or >= deviceCount), the function returns false.
 * Otherwise, it calls cudaSetDevice(id) and returns true only
 * if that call succeeds.
 *
 * @param id The desired CUDA device ID.
 * @return true if the device was successfully set; false otherwise.
 */
bool set_device(int id)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        // Could log the error message if desired:
        // fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (id < 0 || id >= deviceCount) {
        // ID is out of bounds
        return false;
    }

    err = cudaSetDevice(id);
    return (err == cudaSuccess);
}

/**
 * @brief Simple test harness for set_device().
 *
 * This function prints the number of CUDA devices and attempts
 * to set a device. It demonstrates both successful and
 * unsuccessful cases.
 */
int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices available.\n");
        return 0;
    }

    // Attempt to set the first device (ID 0)
    if (set_device(0)) {
        printf("Successfully set device 0.\n");
    } else {
        printf("Failed to set device 0.\n");
    }

    // Attempt to set an invalid device ID (deviceCount + 1)
    int invalid_id = deviceCount + 1;
    if (set_device(invalid_id)) {
        printf("Unexpectedly succeeded in setting device %d.\n", invalid_id);
    } else {
        printf("Correctly failed to set invalid device ID %d.\n", invalid_id);
    }

    return 0;
}
