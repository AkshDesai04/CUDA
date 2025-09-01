/*
Aim of the program: Create a test suite function that deliberately triggers and checks for several different invalid value errors.

Thinking:
The goal is to write a CUDA C program that deliberately performs operations known to produce
`cudaErrorInvalidValue` or other related CUDA error codes.  
We want to cover a variety of typical misuse scenarios:

1. `cudaMalloc` with a size of 0 – should return `cudaErrorInvalidValue`.
2. `cudaMalloc` with a NULL pointer to receive the device address – also `cudaErrorInvalidValue`.
3. `cudaMemcpy` where either the source or destination pointer is NULL – again `cudaErrorInvalidValue`.
4. Launching a kernel with an invalid configuration (zero threads per block) – should return
   `cudaErrorInvalidConfiguration`.
5. Creating an event with an invalid flag value – should return `cudaErrorInvalidValue`.

For each case we will create a small helper function that performs the operation and checks the
return code against the expected error.  The main function will run each test and report
whether it passed or failed.  The code is kept simple and self‑contained, using only the
CUDA runtime API, `stdio.h` for printing, and basic boolean checks.  All error checks are
explicit and the tests deliberately trigger failure conditions, making the suite useful for
debugging error handling logic.

Below is the complete CUDA .cu file implementing this test suite.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdbool.h>

__global__ void dummyKernel(void) {
    // Empty kernel – does nothing
}

bool test_malloc_zero_size(void) {
    void* devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, 0);
    if (err != cudaErrorInvalidValue) {
        printf("test_malloc_zero_size failed: expected cudaErrorInvalidValue (%d), got %d (%s)\n",
               cudaErrorInvalidValue, (int)err, cudaGetErrorString(err));
        return false;
    }
    printf("test_malloc_zero_size passed.\n");
    return true;
}

bool test_malloc_null_ptr(void) {
    cudaError_t err = cudaMalloc(NULL, 1024);
    if (err != cudaErrorInvalidValue) {
        printf("test_malloc_null_ptr failed: expected cudaErrorInvalidValue (%d), got %d (%s)\n",
               cudaErrorInvalidValue, (int)err, cudaGetErrorString(err));
        return false;
    }
    printf("test_malloc_null_ptr passed.\n");
    return true;
}

bool test_memcpy_null_src(void) {
    void* devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    if (err != cudaSuccess) {
        printf("test_memcpy_null_src failed at cudaMalloc: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(NULL, devPtr, 1024, cudaMemcpyDeviceToHost);
    if (err != cudaErrorInvalidValue) {
        printf("test_memcpy_null_src failed: expected cudaErrorInvalidValue (%d), got %d (%s)\n",
               cudaErrorInvalidValue, (int)err, cudaGetErrorString(err));
        cudaFree(devPtr);
        return false;
    }
    cudaFree(devPtr);
    printf("test_memcpy_null_src passed.\n");
    return true;
}

bool test_memcpy_null_dst(void) {
    void* devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    if (err != cudaSuccess) {
        printf("test_memcpy_null_dst failed at cudaMalloc: %s\n", cudaGetErrorString(err));
        return false;
    }
    int hostBuf[256] = {0};
    err = cudaMemcpy(devPtr, NULL, 1024, cudaMemcpyHostToDevice);
    if (err != cudaErrorInvalidValue) {
        printf("test_memcpy_null_dst failed: expected cudaErrorInvalidValue (%d), got %d (%s)\n",
               cudaErrorInvalidValue, (int)err, cudaGetErrorString(err));
        cudaFree(devPtr);
        return false;
    }
    cudaFree(devPtr);
    printf("test_memcpy_null_dst passed.\n");
    return true;
}

bool test_kernel_launch_invalid_config(void) {
    cudaError_t err = cudaLaunchKernel((const void*)dummyKernel,
                                       dim3(0), dim3(1), NULL, 0, NULL);
    // The function signature of cudaLaunchKernel requires a pointer to the kernel function.
    // Passing a nullptr for the function pointer should yield cudaErrorInvalidDeviceFunction.
    if (err != cudaErrorInvalidDeviceFunction) {
        printf("test_kernel_launch_invalid_config failed: expected cudaErrorInvalidDeviceFunction (%d), got %d (%s)\n",
               cudaErrorInvalidDeviceFunction, (int)err, cudaGetErrorString(err));
        return false;
    }
    printf("test_kernel_launch_invalid_config passed.\n");
    return true;
}

bool test_event_create_invalid_flag(void) {
    cudaEvent_t ev;
    cudaError_t err = cudaEventCreateWithFlags(&ev, 0xFFFFFFFF);
    if (err != cudaErrorInvalidValue) {
        printf("test_event_create_invalid_flag failed: expected cudaErrorInvalidValue (%d), got %d (%s)\n",
               cudaErrorInvalidValue, (int)err, cudaGetErrorString(err));
        return false;
    }
    printf("test_event_create_invalid_flag passed.\n");
    return true;
}

int main(void) {
    bool all_passed = true;

    printf("Running CUDA invalid value error test suite...\n\n");

    all_passed &= test_malloc_zero_size();
    all_passed &= test_malloc_null_ptr();
    all_passed &= test_memcpy_null_src();
    all_passed &= test_memcpy_null_dst();
    all_passed &= test_kernel_launch_invalid_config();
    all_passed &= test_event_create_invalid_flag();

    printf("\nTest suite completed. %s\n", all_passed ? "All tests passed." : "Some tests failed.");
    return all_passed ? 0 : 1;
}
