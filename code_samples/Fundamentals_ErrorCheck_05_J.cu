/*
Aim: Trigger `cudaErrorDeviceAlreadyInUse` (this is harder, might require multiple processes).

The CUDA Runtime API can return `cudaErrorDeviceAlreadyInUse` when an operation
such as `cudaDeviceReset` is attempted while the device is still being used by
another process.  To demonstrate this we spawn a child process that acquires a
CUDA context on the default device (device 0) and launches an infinite loop
kernel that never completes.  This keeps the context alive.  The parent process
waits a short time to allow the child to launch the kernel, then calls
`cudaDeviceReset`.  Because the child still owns the context, `cudaDeviceReset`
fails and returns `cudaErrorDeviceAlreadyInUse`.  We capture and print this
error to verify that it occurred.

The child never exits; it simply blocks forever in the infinite loop kernel,
which means its CUDA context remains active until the process is killed.
The parent waits for the child to finish (after printing the error) before
exiting cleanly.

Implementation notes:
- We use the runtime API (`cudaSetDevice`, `cudaDeviceReset`) for simplicity.
- The kernel is a simple infinite loop: `while(true) {}`.  It will consume a
  GPU thread until the process is terminated.
- We use `fork()` to create a child process, and `waitpid()` to wait for the
  child in the parent.
- The code prints the error string for clarity.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

// Simple kernel that never terminates
__global__ void infiniteLoopKernel()
{
    while (true) {}
}

int main()
{
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return EXIT_FAILURE;
    } else if (pid == 0) {
        /* Child process */
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Child: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        /* Launch the infinite loop kernel on the default stream.
           The kernel will never complete, keeping the context alive. */
        infiniteLoopKernel<<<1, 1>>>();
        /* We don't call cudaDeviceSynchronize() because the kernel never returns.
           The process will simply block on the kernel launch, keeping the
           context alive for the parent to attempt a reset. */

        /* In practice, we should never reach this point. */
        fprintf(stderr, "Child: Unexpectedly reached end of main.\n");
        exit(EXIT_FAILURE);
    } else {
        /* Parent process */
        /* Give the child a moment to launch the kernel */
        sleep(1);

        /* Attempt to reset the device while the child holds a context. */
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            printf("Parent: cudaDeviceReset failed with error: %s (%d)\n",
                   cudaGetErrorString(err), err);
        } else {
            printf("Parent: cudaDeviceReset succeeded.\n");
        }

        /* Wait for the child process to finish (it won't unless killed). */
        int status;
        waitpid(pid, &status, 0);

        return EXIT_SUCCESS;
    }
}
