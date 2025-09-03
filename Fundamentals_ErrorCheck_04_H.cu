```cu
/*
Aim of the program:
Create a test suite function that deliberately triggers and checks for several different invalid value errors.

Thinking:
- The goal is to produce a self-contained CUDA program that exercises a variety of common
  CUDA error conditions that arise from "invalid value" misuse.  These include:
    1. Launching a kernel with an invalid configuration (zero threads per block).
    2. Requesting an excessively large memory allocation that will fail.
    3. Attempting to use an invalid device index when setting the active device.
    4. Performing a memory copy with a NULL pointer or mismatched sizes.
- For each of these error scenarios we:
    * Perform the operation that should fail.
    * Call `cudaGetLastError()` immediately to capture the error code.
    * Print a message indicating whether the expected error was observed.
- The program defines a very simple kernel (`dummyKernel`) which does nothing; it is only
  used to trigger the launch configuration error.
- The `testInvalidValueErrors()` function encapsulates all the checks and reports the
  outcome of each test case.
- Finally, `main()` simply calls the test suite and exits.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Very simple kernel that does nothing; used only for launch configuration tests.
__global__ void dummyKernel()
{
    // No operation
}

// Helper macro to check CUDA error and print message
#define CHECK_CUDA_ERR(err, msg)                          \
    do {                                                 \
        cudaError_t err_code = (err);                    \
        if (err_code != cudaSuccess) {                   \
            printf("[ERROR] %s: %s\n", msg, cudaGetErrorString(err_code)); \
        } else {                                         \
            printf("[INFO] %s succeeded.\n", msg);        \
        }                                                \
    } while(0)

// Test 1: Launch kernel with zero threads per block (invalid configuration)
void testKernelLaunchInvalidConfig()
{
    printf("\n[Test 1] Kernel launch with zero threads per block...\n");
    dummyKernel<<<1, 0>>>();
    cudaError_t err = cudaGetLastError();
    if (err == cudaErrorInvalidConfiguration) {
        printf("[PASS] Expected error caught: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error: %s\n", cudaGetErrorString(err));
    }
}

// Test 2: Request excessively large memory allocation (invalid value)
void testMallocInvalidSize()
{
    printf("\n[Test 2] cudaMalloc with excessively large size...\n");
    size_t hugeSize = (size_t)4 * 1024 * 1024 * 1024 * 1024; // 4 TB
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, hugeSize);
    if (err == cudaErrorMemoryAllocation || err == cudaErrorInvalidValue) {
        printf("[PASS] Expected error caught: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error: %s\n", cudaGetErrorString(err));
    }
    if (ptr) cudaFree(ptr); // cleanup if somehow allocated
}

// Test 3: Set device to an invalid index (invalid device)
void testSetInvalidDevice()
{
    printf("\n[Test 3] cudaSetDevice with invalid index...\n");
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int invalidIndex = deviceCount + 10; // out of range
    cudaError_t err = cudaSetDevice(invalidIndex);
    if (err == cudaErrorInvalidDevice) {
        printf("[PASS] Expected error caught: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error: %s\n", cudaGetErrorString(err));
    }
}

// Test 4: cudaMemcpy with NULL source or destination pointer (invalid value)
void testMemcpyInvalidPointers()
{
    printf("\n[Test 4] cudaMemcpy with NULL pointers...\n");
    int hostData = 42;
    int* devPtr = NULL;
    cudaError_t err;

    // Attempt to copy from NULL device pointer to host
    err = cudaMemcpy(&hostData, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaErrorInvalidValue) {
        printf("[PASS] Expected error caught for devToHost: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error for devToHost: %s\n", cudaGetErrorString(err));
    }

    // Allocate valid device memory for further test
    cudaMalloc(&devPtr, sizeof(int));
    // Attempt to copy from device to NULL host pointer
    err = cudaMemcpy(NULL, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaErrorInvalidValue) {
        printf("[PASS] Expected error caught for hostToDev: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error for hostToDev: %s\n", cudaGetErrorString(err));
    }

    cudaFree(devPtr);
}

// Test 5: Launch kernel with more blocks than the maximum allowed (invalid value)
void testKernelLaunchExceedMaxBlocks()
{
    printf("\n[Test 5] Kernel launch with exceeding maximum number of blocks...\n");
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Determine max grid dimension in x (maximum allowed blocks)
    int maxBlocks = prop.maxGridSize[0];
    // Try to launch with one more block than allowed
    dummyKernel<<<(dim3)(maxBlocks + 1), 256>>>();
    cudaError_t err = cudaGetLastError();
    if (err == cudaErrorInvalidConfiguration || err == cudaErrorLaunchFailure) {
        printf("[PASS] Expected error caught: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error: %s\n", cudaGetErrorString(err));
    }
}

// Test 6: Attempt to synchronize an invalid stream (invalid value)
void testStreamSyncInvalid()
{
    printf("\n[Test 6] cudaStreamSynchronize with invalid stream...\n");
    cudaStream_t invalidStream = (cudaStream_t)0xdeadbeef;
    cudaError_t err = cudaStreamSynchronize(invalidStream);
    if (err == cudaErrorInvalidValue) {
        printf("[PASS] Expected error caught: %s\n", cudaGetErrorString(err));
    } else {
        printf("[FAIL] Unexpected error: %s\n", cudaGetErrorString(err));
    }
}

// Main test suite that runs all tests
void testInvalidValueErrors()
{
    testKernelLaunchInvalidConfig();
    testMallocInvalidSize();
    testSetInvalidDevice();
    testMemcpyInvalidPointers();
    testKernelLaunchExceedMaxBlocks();
    testStreamSyncInvalid();
}

int main()
{
    printf("=== CUDA Invalid Value Error Test Suite ===\n");
    testInvalidValueErrors();
    printf("\nAll tests completed.\n");
    return 0;
}
```