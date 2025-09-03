Below is a typical “leaky” CUDA C/C++ program (it allocates device memory and never frees it).  
Running it with  

```
cuda-memcheck --leak-check full ./leaky_cuda
```  

produces an output that reports the missing de‑allocation.  
Because the program cannot actually be executed here, the output shown is a **representative example** that you would normally see on a system that has a GPU and `cuda-memcheck` installed.

---

## Example “leaky” program (leaky_cuda.cu)

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = idx * 1.0f;
}

int main(void)
{
    int N = 1024;
    float *d_arr = NULL;

    // Allocate 4*N bytes on the device (float = 4 bytes)
    cudaMalloc((void**)&d_arr, N * sizeof(float));

    // Launch kernel
    kernel<<<(N+255)/256, 256>>>(d_arr, N);
    cudaDeviceSynchronize();

    // **BUG**: d_arr is never freed (cudaFree is missing)

    return 0;
}
```

Compile with

```
nvcc -o leaky_cuda leaky_cuda.cu
```

---

## Sample `cuda-memcheck` output

```
==12345== cuda-memcheck, Version 12.2
==12345== Copyright (c) 2021, NVIDIA Corporation
==12345== 
==12345== --leak-check full
==12345== ------------------------------------------------------------------
==12345== Device Memory Leak Summary
==12345== ------------------------------------------------------------------
==12345== 4,096 bytes in 1 allocation(s) are still reachable at exit
==12345== 
==12345==   at 0x1004B9C (malloc @ 0x7ff6d5b3e8d0)
==12345==   by 0x5A2B8 (cudaMalloc @ /usr/local/cuda/lib64/libcudart.so.12)
==12345==   by 0x55A1F (kernel @ leaky_cuda.cu:15)
==12345== 
==12345== LEAK SUMMARY
==12345==   definitely lost:   0 bytes in 0 blocks
==12345==   indirectly lost:  0 bytes in 0 blocks
==12345==   possibly lost:    0 bytes in 0 blocks
==12345==   still reachable: 4,096 bytes in 1 blocks
==12345==   suppressed:       0 bytes in 0 blocks
==12345== 
==12345== ERROR SUMMARY: 0 errors from 0 contexts
```

---

## How to read the output

| Section | Meaning |
|---------|---------|
| **Device Memory Leak Summary** | `cuda-memcheck` lists each device‑memory allocation that was still alive when the program terminated. Here it reports **4,096 bytes in 1 allocation** that were *still reachable* (i.e., the pointer existed but was never freed). |
| **Traceback** (`at ...`) | Shows the call stack that created the leaked allocation. `cudaMalloc` is called inside the kernel (via a driver call), and the stack points to the line in `leaky_cuda.cu` where `cudaMalloc` was invoked. |
| **LEAK SUMMARY** | Classic Valgrind‑style categories: |
|  | *definitely lost* – memory you cannot recover from any pointer. |
|  | *indirectly lost* – memory that was pointed to by a lost block. |
|  | *possibly lost* – memory that may be recoverable depending on pointer analysis. |
|  | *still reachable* – memory that is still reachable via a pointer but was never freed (exactly what we have). |
|  | *suppressed* – memory intentionally not checked. |
| **ERROR SUMMARY** | `0 errors from 0 contexts` indicates that, apart from the leak, the program did not touch invalid device memory, hit a segmentation fault, etc. |

---

## What the leak means

The allocation `cudaMalloc((void**)&d_arr, N * sizeof(float));` was never paired with a `cudaFree(d_arr);`. When the program ends, CUDA’s memory manager keeps the allocation alive (so the device can still use it, and the driver does not have to immediately reclaim it). `cuda-memcheck` warns you that this is a potential resource leak, especially in long‑running or iterative applications where many such allocations could exhaust GPU memory.

---

## Fixing the leak

Add a single line before `return 0;`:

```cpp
cudaFree(d_arr);
```

Re‑compile and run `cuda-memcheck` again. The output should now report **no leaked allocations**:

```
==12345== Device Memory Leak Summary
==12345==   0 bytes in 0 allocation(s) are still reachable at exit
...
==12345== LEAK SUMMARY
==12345==   definitely lost:   0 bytes in 0 blocks
==12345==   ...
==12345== ERROR SUMMARY: 0 errors from 0 contexts
```

---

## Take‑away

* `cuda-memcheck --leak-check full` is the tool of choice for discovering device‑memory leaks.
* The “still reachable” section identifies allocations that survive to program exit but were never freed.
* The traceback helps you pinpoint the exact line of code where the allocation was made.
* Always pair each `cudaMalloc` (or `cudaMallocManaged`) with a corresponding `cudaFree` (or `cudaFreeManaged`) to keep the device memory usage bounded.