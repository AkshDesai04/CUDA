/*
Aim: Add detailed comments to your two code snippets explaining exactly why one error is synchronous and the other is asynchronous.

Thinking:
- In CUDA, error handling differs between operations that are performed synchronously on the host (e.g., cudaMemcpy, cudaMalloc) and those that are queued on the device for asynchronous execution (e.g., kernel launches, cudaMemcpyAsync).
- A **synchronous** error occurs when the function call itself blocks the host until the operation is completed or fails. The error is reported immediately by the function return value. Typical examples include cudaMalloc and cudaMemcpy where the host thread is blocked until the memory allocation or transfer finishes.
- An **asynchronous** error occurs when the function call only queues the operation in a stream and returns immediately. The host thread is not blocked; the actual execution happens on the GPU. Errors in such operations are not reported until a synchronization point is reached or cudaGetLastError() is explicitly called. Kernel launches and cudaMemcpyAsync are common examples.

The program below demonstrates two scenarios:
1. A synchronous error: calling cudaMemcpy with an invalid host pointer. The error is returned immediately by the function.
2. An asynchronous error: launching a kernel with an invalid configuration (too many blocks) that causes the launch to fail. The error is only detected when we synchronize with cudaDeviceSynchronize() or explicitly call cudaGetLastError().

Both snippets contain comprehensive comments explaining why the error is synchronous or asynchronous and how CUDA reports it.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Helper macro to check CUDA errors synchronously. It prints the
 * error message and exits if the error code is not cudaSuccess.
 */
#define CHECK_CUDA_CALL(call)                                       \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* ------------------------------------------------------------------ */
/* Synchronous error example:
 *  - We allocate a small buffer on the host.
 *  - We attempt to copy data from device to a NULL host pointer.
 *  - cudaMemcpy is a blocking call; it will attempt to copy and
 *    immediately detect that the destination pointer is invalid.
 *  - The error code (cudaErrorInvalidValue) is returned by cudaMemcpy
 *    before the function returns, making the error synchronous.
 */
void synchronous_error_example()
{
    // Allocate a tiny device buffer
    int *d_data;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_data, sizeof(int)));

    // Fill it with some value
    int val = 42;
    CHECK_CUDA_CALL(cudaMemcpy(d_data, &val, sizeof(int), cudaMemcpyHostToDevice));

    // Intentionally pass a NULL pointer as destination host address
    int *h_invalid = NULL;

    // This call will fail synchronously because the host pointer is NULL.
    // The error is reported by the return value immediately.
    cudaError_t err = cudaMemcpy(h_invalid, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // Because cudaMemcpy is synchronous, we can catch the error right away.
        printf("Synchronous error detected: %s\n", cudaGetErrorString(err));
    } else {
        printf("Unexpected success (this should not happen)\n");
    }

    // Clean up
    cudaFree(d_data);
}

/* ------------------------------------------------------------------ */
/* Asynchronous error example:
 *  - We launch a trivial kernel with an intentionally incorrect
 *    launch configuration: too many blocks to exceed the device's
 *    maximum grid size.
 *  - Kernel launch is an asynchronous operation; it queues the kernel
 *    in the default stream and returns immediately.
 *  - Because the launch parameters are invalid, the kernel launch
 *    fails, but the error is not reported until we call
 *    cudaGetLastError() or synchronize the stream.
 *  - The error is therefore asynchronous: the failure is hidden
 *    until a synchronization point.
 */
__global__ void trivialKernel()
{
    // Kernel body is empty; we only care about the launch.
}

void asynchronous_error_example()
{
    // Determine maximum grid size for this device
    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    // Intentionally use an invalid grid dimension: one dimension larger than the limit
    dim3 invalidGrid(prop.maxGridSize[0] + 1, 1, 1);
    dim3 blockDim(1, 1, 1);

    // Launch the kernel with invalid grid size. This call is asynchronous.
    trivialKernel<<<invalidGrid, blockDim>>>();

    // At this point, the kernel launch has been queued and the function has returned.
    // No error is reported yet because the launch was asynchronous.

    // We can explicitly query the error using cudaGetLastError()
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The error is detected asynchronously; it appears only after we query.
        printf("Asynchronous error detected by cudaGetLastError: %s\n", cudaGetErrorString(err));
    } else {
        // If no error was detected, we still need to synchronize to catch any
        // errors that may occur during execution.
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        printf("Kernel executed successfully (this should not happen with invalid grid size)\n");
    }
}

/* ------------------------------------------------------------------ */
int main()
{
    printf("=== Synchronous Error Example ===\n");
    synchronous_error_example();

    printf("\n=== Asynchronous Error Example ===\n");
    asynchronous_error_example();

    return 0;
}
