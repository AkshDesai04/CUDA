/*
 * Aim of the program (as passed in the prompt):
 * Does the leaked memory get freed when the application exits? Use `nvidia-smi` to confirm.
 *
 * Thinking behind the implementation:
 * 1. We want to create a small CUDA program that allocates device memory but intentionally
 *    does not free it, thereby creating a "leak" from the perspective of the program.
 * 2. In CUDA, the driver automatically releases all allocated resources when a process
 *    terminates. This includes memory allocated with cudaMalloc. Therefore, after the
 *    program exits, the GPU should report that the memory has been freed.
 * 3. To demonstrate this, the program will:
 *    - Allocate a fixed amount of device memory (e.g., 100 MB).
 *    - Print a message indicating the allocation succeeded.
 *    - Sleep for a short period (5 seconds) to allow the user to run `nvidia-smi` while
 *      the program is still running.
 *    - Exit without calling cudaFree on the allocated memory.
 * 4. After the program terminates, the user can run `nvidia-smi` to observe that the
 *    memory used by the program is no longer allocated.
 * 5. The code is written in C/C++ (CUDA C) and should compile with `nvcc`.
 * 6. No external libraries other than the CUDA runtime and standard C++ headers are
 *    required.
 *
 * Notes:
 * - The program does not launch any kernel; the goal is purely to allocate memory.
 * - If the allocation fails, the program reports the error and exits.
 * - The sleep duration can be adjusted as needed.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>   // for sleep
#include <cstdlib>   // for EXIT_FAILURE, EXIT_SUCCESS

int main() {
    const size_t allocSize = 100 * 1024 * 1024; // 100 MB
    void* d_ptr = nullptr;

    cudaError_t err = cudaMalloc(&d_ptr, allocSize);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Allocated " << allocSize / (1024 * 1024) << " MB on device." << std::endl;
    std::cout << "Sleeping for 5 seconds. While this program runs, you can run 'nvidia-smi' "
                 "to observe the allocated memory." << std::endl;

    sleep(5);

    std::cout << "Exiting without freeing the allocated memory." << std::endl;
    // Intentionally not calling cudaFree(d_ptr)

    return EXIT_SUCCESS;
}
