```cpp
/*
Aim of the program: Write a version of the function for streams: `checkStream(cudaStream_t stream)` which calls `cudaStreamSynchronize`.

Thinking:
- The task is to provide a CUDA helper function that synchronizes a given CUDA stream.
- We need to include the necessary CUDA headers (`<cuda_runtime.h>`) and standard I/O for error reporting (`<cstdio>`).
- The function `checkStream` should take a `cudaStream_t` as its argument.
- Inside the function, call `cudaStreamSynchronize(stream)` and capture the returned `cudaError_t`.
- If the returned error code is not `cudaSuccess`, print an error message to `stderr` using `cudaGetErrorString`.
- The function does not need to return a value; it performs the synchronization and reports errors if any.
- This file is a complete `.cu` file ready to be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <cstdio>

// Synchronizes the specified CUDA stream and reports errors if they occur.
void checkStream(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing stream: %s\n", cudaGetErrorString(err));
    }
}
```