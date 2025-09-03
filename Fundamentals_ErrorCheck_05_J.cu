/*
Trigger `cudaErrorDeviceAlreadyInUse` (this is harder, might require multiple processes).

The goal of this program is to deliberately provoke the CUDA error
`cudaErrorDeviceAlreadyInUse`, which occurs when two separate
processes attempt to create a CUDA context on the same GPU device
simultaneously.  To achieve this, the program forks itself into a
parent and a child process.  The parent immediately creates a CUDA
context on device 0 and launches a long‑running dummy kernel that
keeps the device busy.  It then sleeps for a few seconds to give the
child a chance to run.  The child, after a short delay, attempts to
create its own CUDA context on the same device.  Since the parent
already owns the device, the child’s `cudaSetDevice(0)` call fails
with `cudaErrorDeviceAlreadyInUse`.  The program reports the error
string to demonstrate that the error was indeed triggered.

Key points considered:
- CUDA contexts are per‑process; after `fork()`, the child does **not**
  inherit the parent’s context, so it will try to create a new one.
- Launching a kernel asynchronously (without `cudaDeviceSynchronize`)
  ensures the device remains in use during the child’s attempt.
- `sleep()` is used to coordinate timing between the two processes.
- The dummy kernel performs a long but finite loop to keep the GPU
  busy without causing an infinite hang.
- Error handling prints the specific CUDA error code and message
  from `cudaGetErrorString`.

Compile with: `nvcc -o trigger_error trigger_error.cu`
Run: `./trigger_error`
*/

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/wait.h>

__global__ void busyKernel()
{
    // Simple busy loop to keep the GPU busy for a while
    for (long long i = 0; i < 1000000000LL; ++i) {
        // Prevent optimization
        if (i == 0x12345678) {
            break;
        }
    }
}

int main()
{
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) { // Child process
        // Give parent a chance to set up the device
        sleep(1);

        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            printf("Child: cudaSetDevice(0) returned error: %s\n",
                   cudaGetErrorString(err));
        } else {
            // If somehow succeeded, attempt a simple allocation to force context
            int *d_ptr = NULL;
            err = cudaMalloc((void**)&d_ptr, 1024);
            if (err != cudaSuccess) {
                printf("Child: cudaMalloc returned error: %s\n",
                       cudaGetErrorString(err));
            } else {
                printf("Child: cudaMalloc succeeded unexpectedly.\n");
                cudaFree(d_ptr);
            }
        }
        return 0;
    } else { // Parent process
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            printf("Parent: cudaSetDevice(0) returned error: %s\n",
                   cudaGetErrorString(err));
            return 1;
        }

        // Launch a long running kernel asynchronously
        busyKernel<<<1, 1>>>();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Parent: kernel launch error: %s\n",
                   cudaGetErrorString(err));
        }

        // Keep the device busy and allow child to attempt device use
        sleep(5);

        // Wait for child to finish
        wait(NULL);
        return 0;
    }
}
