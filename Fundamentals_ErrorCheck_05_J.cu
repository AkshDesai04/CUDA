```cu
/*
Aim: Trigger cudaErrorDeviceAlreadyInUse (this is harder, might require multiple processes).

Thinking:
The cudaErrorDeviceAlreadyInUse error occurs when a CUDA device already has an active
context that cannot be shared with another context from the same or another process
(when Multi‑Process Service is not enabled).  The simplest way to provoke this error
is to create a CUDA context in one process and then spawn a second process that
attempts to create its own context on the same device while the first one is still
alive.  By using `fork()` we can create a child process that inherits the parent’s
process ID but starts with no CUDA context of its own.  The child will then call
`cudaFree(0)` (which lazily creates a new context) and immediately launch a tiny
kernel.  Because the parent process still holds the first context, the runtime
will reject the second context creation and return cudaErrorDeviceAlreadyInUse.
The child process will capture and print this error.  The parent waits for the child
to finish and then exits, leaving the first context alive during the child’s
execution to guarantee the conflict.
*/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

// Simple dummy kernel
__global__ void dummyKernel() { /* does nothing */ }

int main() {
    // Create first context in parent
    cudaError_t err = cudaFree(0); // lazy context creation
    if (err != cudaSuccess) {
        fprintf(stderr, "Parent: cudaFree(0) failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch a dummy kernel to ensure context is active
    dummyKernel<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Parent: kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Parent: device sync failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Keep parent context alive for a moment
    sleep(1);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return EXIT_FAILURE;
    }

    if (pid == 0) {
        // Child process
        // Attempt to create a new context on the same device
        err = cudaFree(0); // Should try to create a new context
        if (err != cudaSuccess) {
            printf("Child: cudaFree(0) error: %d (%s)\n", err, cudaGetErrorString(err));
        } else {
            // Launch kernel to trigger potential conflict
            dummyKernel<<<1, 1>>>();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Child: kernel launch error: %d (%s)\n", err, cudaGetErrorString(err));
            } else {
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("Child: device sync error: %d (%s)\n", err, cudaGetErrorString(err));
                } else {
                    printf("Child: kernel executed successfully (unexpected)\n");
                }
            }
        }
        return 0;
    } else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            printf("Parent: child exited with status %d\n", WEXITSTATUS(status));
        } else {
            printf("Parent: child did not exit cleanly\n");
        }
        return 0;
    }
}
```