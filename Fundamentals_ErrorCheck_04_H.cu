/*
Create a test suite function that deliberately triggers and checks for several different invalid value errors.

My thoughts:
The goal is to produce a small CUDA program that exercises various CUDA API calls with intentionally bad arguments
so that the CUDA runtime returns cudaErrorInvalidValue.  The test suite will call several functions:
- cudaMalloc with size 0 (should fail).
- cudaMalloc with a null pointer for the device pointer argument (should fail).
- cudaMemcpy with a null destination (should fail).
- cudaMemcpy with a null source (should fail).
- Kernel launch with an illegal grid or block dimension (should fail).

For each case we capture the returned error code and compare it to the expected
cudaErrorInvalidValue.  The results are printed to stdout.  After the test suite runs,
the program exits with a success code.  The code is self‑contained in a single .cu file
and uses only standard CUDA runtime API functions.  No external libraries are required.

To compile this code one would use:
    nvcc -arch=sm_70 -o test_invalid_value test_invalid_value.cu
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple kernel used for launch tests */
__global__ void dummyKernel()
{
    /* No operation */
}

/* Helper function to compare the error code returned by a CUDA call
   to the expected error code and print a pass/fail message. */
void checkError(cudaError_t err,
                const char *testName,
                cudaError_t expected)
{
    if (err == expected)
    {
        printf("[PASS] %s returned expected error: %s\n",
               testName, cudaGetErrorString(err));
    }
    else
    {
        printf("[FAIL] %s returned %s, expected %s\n",
               testName, cudaGetErrorString(err), cudaGetErrorString(expected));
    }
}

/* Test 1: cudaMalloc with size 0 */
void testCudaMallocZero()
{
    int *devPtr = NULL;
    cudaError_t err = cudaMalloc((void**)&devPtr, 0);
    checkError(err, "cudaMalloc(0)", cudaErrorInvalidValue);
}

/* Test 2: cudaMalloc with null pointer for device pointer argument */
void testCudaMallocNullPtr()
{
    /* Passing a null pointer to the first argument of cudaMalloc */
    cudaError_t err = cudaMalloc(NULL, 1024);
    checkError(err, "cudaMalloc(NULL)", cudaErrorInvalidValue);
}

/* Test 3: cudaMemcpy with null destination */
void testCudaMemcpyNullDst()
{
    int *devPtr = NULL;
    /* Allocate device memory for a valid test of source pointer */
    cudaError_t err = cudaMalloc((void**)&devPtr, 1024);
    if (err != cudaSuccess)
    {
        printf("[FAIL] cudaMalloc for testCudaMemcpyNullDst failed: %s\n",
               cudaGetErrorString(err));
        return;
    }

    /* Now attempt to copy to a null destination */
    err = cudaMemcpy(NULL, devPtr, 1024, cudaMemcpyDeviceToHost);
    checkError(err, "cudaMemcpy(NULL dest)", cudaErrorInvalidValue);

    /* Clean up */
    cudaFree(devPtr);
}

/* Test 4: cudaMemcpy with null source */
void testCudaMemcpyNullSrc()
{
    int *devPtr = NULL;
    /* Allocate device memory for a valid test of destination pointer */
    cudaError_t err = cudaMalloc((void**)&devPtr, 1024);
    if (err != cudaSuccess)
    {
        printf("[FAIL] cudaMalloc for testCudaMemcpyNullSrc failed: %s\n",
               cudaGetErrorString(err));
        return;
    }

    /* Now attempt to copy from a null source */
    err = cudaMemcpy(devPtr, NULL, 1024, cudaMemcpyHostToDevice);
    checkError(err, "cudaMemcpy(NULL src)", cudaErrorInvalidValue);

    /* Clean up */
    cudaFree(devPtr);
}

/* Test 5: Kernel launch with invalid grid or block dimensions */
void testKernelLaunchInvalid()
{
    /* Launch with zero grid dimensions */
    dummyKernel<<<dim3(0, 1, 1), dim3(1, 1, 1)>>>();
    cudaError_t err = cudaGetLastError();
    checkError(err, "kernel launch with zero grid dims", cudaErrorInvalidValue);

    /* Launch with zero block dimensions */
    dummyKernel<<<dim3(1, 1, 1), dim3(0, 1, 1)>>>();
    err = cudaGetLastError();
    checkError(err, "kernel launch with zero block dims", cudaErrorInvalidValue);
}

/* Main test suite that runs all individual tests */
int main(void)
{
    /* Ensure we are on a device */
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("[ERROR] No CUDA devices found.\n");
        return 1;
    }
    cudaSetDevice(0);

    printf("Running CUDA invalid value error test suite...\n\n");

    testCudaMallocZero();
    testCudaMallocNullPtr();
    testCudaMemcpyNullDst();
    testCudaMemcpyNullSrc();
    testKernelLaunchInvalid();

    printf("\nTest suite completed.\n");
    return 0;
}
