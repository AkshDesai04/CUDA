The **“sticky” error state** in CUDA refers to a situation where an error that occurs during a kernel launch or a CUDA API call is recorded and remains “sticky” until it is explicitly cleared or the device is reset. Once a sticky error is set, subsequent CUDA calls will return that same error, even if the underlying problem has been fixed or the offending code has been corrected. This mechanism is designed to prevent the program from continuing to execute in an invalid or unstable state, which could lead to undefined behavior or data corruption.

### Key points about the sticky error state

| Aspect | Description |
|--------|-------------|
| **When it occurs** | After a kernel launch or a CUDA runtime API call that fails (e.g., due to invalid memory access, illegal instruction, out‑of‑bounds index, etc.). |
| **What it records** | The error code (e.g., `cudaErrorIllegalAddress`, `cudaErrorLaunchFailure`, etc.) returned by the failing call. |
| **Persistence** | The error remains in the device’s error state until one of the following actions is taken: <br>1. The user explicitly clears it using `cudaGetLastError()` or `cudaPeekAtLastError()`. <br>2. The device is reset with `cudaDeviceReset()` or the context is destroyed. |
| **Effect on subsequent calls** | Any subsequent CUDA API call will return the same error code, regardless of whether the operation would normally succeed. This prevents silent propagation of errors. |
| **Why it’s useful** | It forces developers to detect and handle errors promptly. It also helps debugging by ensuring that errors are not ignored and that the program doesn’t continue in a corrupted state. |
| **How to handle it** | 1. **Check error after every call**: `cudaError_t err = cudaFunc(); if (err != cudaSuccess) { /* handle */ }`. <br>2. **Clear the state**: `cudaGetLastError();` after handling to reset the sticky error. |
| **Common scenarios** | * Illegal memory accesses within a kernel (e.g., dereferencing a NULL pointer). <br>* Launching a kernel with insufficient grid or block dimensions. <br>* Using an invalid device pointer or a freed pointer. <br>* Running out of device memory during a kernel launch. |

### Practical example

```cpp
__global__ void kernel(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_arr[idx] = idx;        // normal operation
    } else {
        d_arr[idx] = 0;          // out‑of‑bounds write -> illegal address
    }
}

int main() {
    int *d_arr;
    cudaMalloc(&d_arr, 10 * sizeof(int));

    kernel<<<1, 20>>>(d_arr, 10);   // 20 > 10 => out‑of‑bounds accesses
    cudaError_t err = cudaGetLastError(); // sticky error set to cudaErrorIllegalAddress
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        // clear the sticky error state
        cudaGetLastError(); // resets to cudaSuccess
    }

    // Subsequent calls will succeed because the error was cleared
    cudaMemset(d_arr, 0, 10 * sizeof(int));
    cudaFree(d_arr);
    return 0;
}
```

In this example, the kernel performs an out‑of‑bounds write, causing `cudaErrorIllegalAddress`. This error is sticky: the call to `cudaGetLastError()` returns the error, and until it’s cleared, any further CUDA call would return the same error. By calling `cudaGetLastError()` again (or `cudaPeekAtLastError()`), we reset the error state to `cudaSuccess`, allowing the program to continue normally.

### Summary

The sticky error state is a safety feature that ensures CUDA applications do not silently ignore failures. It compels developers to detect errors early, handle them appropriately, and clear the state before continuing execution. By consistently checking for errors and resetting the sticky state when necessary, you can build robust CUDA programs that behave predictably even in the presence of runtime faults.