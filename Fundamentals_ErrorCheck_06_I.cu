/*
Aim: Write a version of the function for streams: `checkStream(cudaStream_t stream)` which calls `cudaStreamSynchronize`.

Thinking: 
- The goal is to provide a small CUDA helper that synchronizes a given CUDA stream and reports any errors that occur during the synchronization. 
- The function will call `cudaStreamSynchronize(stream)`, capture the returned `cudaError_t`, and if it is not `cudaSuccess`, print an error message and terminate the program. 
- For consistency with typical CUDA error‑checking conventions, we also provide a generic `check()` function that handles any `cudaError_t` and prints a message with the file and line number. 
- A simple `main()` function demonstrates usage: it creates a CUDA stream, performs a trivial kernel launch (to have something running on the stream), and then calls `checkStream()` to wait for completion and verify success. 
- The code uses the CUDA runtime API (`#include <cuda_runtime.h>`) and is written in C with CUDA extensions, compiling as a .cu file. 
- The entire file is self‑contained: includes, helper functions, a trivial kernel, and a `main()` that exercises the `checkStream` wrapper.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper function to check CUDA error codes */
static void check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s:%d]: %s (%d)\n",
                file, line, cudaGetErrorString(err), err);
        exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), __FILE__, __LINE__)

/* Wrapper that synchronizes a stream and checks for errors */
static void checkStream(cudaStream_t stream)
{
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* A trivial kernel that does nothing but will run on the stream */
__global__ void dummyKernel()
{
    /* Empty kernel body */
}

int main(void)
{
    /* Create a CUDA stream */
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    /* Launch a trivial kernel on the stream */
    dummyKernel<<<1, 1, 0, stream>>>();
    CHECK(cudaGetLastError());  /* check kernel launch */

    /* Synchronize the stream using our wrapper */
    checkStream(stream);

    /* Destroy the stream */
    CHECK(cudaStreamDestroy(stream));

    printf("Stream synchronized successfully.\n");
    return 0;
}
