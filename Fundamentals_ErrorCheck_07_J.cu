```cuda
/* ---------------------------------------------------------------
   Snippet 1 – Synchronous error example
   ---------------------------------------------------------------
   This program intentionally launches a kernel with an invalid
   configuration: it requests 1024 threads per block while the
   device supports a maximum of 512.  The CUDA runtime detects
   this mismatch *before* the kernel is queued for execution
   and therefore reports the error synchronously.

   The error becomes visible immediately after the launch
   statement because the runtime performs the configuration
   validation at that point.  No device code has started to
   run, so the program can check cudaGetLastError() right
   after the launch and react to the failure.

   Note that the error is not delayed by a later cudaMemcpy
   or cudaDeviceSynchronize because the failure occurs at
   kernel launch time, not during kernel execution.
*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_out)
{
    int idx = threadIdx.x;
    d_out[idx] = idx;
}

int main(void)
{
    const int N = 1024;
    int *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(int));

    /* Intentionally use too many threads: 1024 > max 512 on most GPUs */
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(1, 1, 1);

    /* Synchronous error: launch configuration is invalid */
    dummyKernel<<<gridDim, blockDim>>>(d_out);

    /* Immediate error check – will catch the launch failure */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Synchronous error (invalid launch config): %s\n",
                cudaGetErrorString(err));
        /* Cleanup */
        cudaFree(d_out);
        return EXIT_FAILURE;
    }

    /* Normally we would do something with the results here. */
    cudaFree(d_out);
    return EXIT_SUCCESS;
}
```

```cuda
/* ---------------------------------------------------------------
   Snippet 2 – Asynchronous error example
   ---------------------------------------------------------------
   In this program we launch a kernel that deliberately performs
   out‑of‑bounds memory accesses.  The kernel is launched with a
   valid configuration, so the launch itself succeeds.  The
   CUDA runtime does not know about the out‑of‑bounds write until
   the kernel executes on the device.  Because kernel execution
   is asynchronous with respect to the host, the error is not
   reported immediately after the launch.  The error becomes
   visible only after a blocking call such as cudaDeviceSynchronize()
   or cudaMemcpy, which forces the host to wait for the device
   to finish its work.  This demonstrates an asynchronous error.

   The key point is that the runtime can only detect memory
   violations after the kernel has run, not at launch time.
   Therefore the host must explicitly synchronize to obtain
   error information.
*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void outOfBoundsKernel(int *d_arr, int size)
{
    int idx = threadIdx.x;
    /* Intentionally write past the end of the array */
    d_arr[idx] = idx;
}

int main(void)
{
    const int N = 256;                // Array size
    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    /* Valid launch configuration: 256 threads, which is <= max threads */
    dim3 blockDim(N, 1, 1);
    dim3 gridDim(1, 1, 1);

    /* Asynchronous launch – will succeed even though kernel will
       write out of bounds.  No error is reported yet. */
    outOfBoundsKernel<<<gridDim, blockDim>>>(d_arr, N);

    /* At this point, the kernel is queued but not yet finished.
       The program proceeds without noticing the out-of-bounds write. */

    /* Force synchronization to wait for kernel completion */
    cudaError_t err = cudaDeviceSynchronize();

    /* Now the error is reported because the device has finished
       executing the kernel and the runtime detected the invalid
       memory access. */
    if (err != cudaSuccess) {
        fprintf(stderr, "Asynchronous error (out-of-bounds memory access): %s\n",
                cudaGetErrorString(err));
    } else {
        printf("Kernel executed without detecting out-of-bounds error.\n");
    }

    cudaFree(d_arr);
    return (err == cudaSuccess) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```
