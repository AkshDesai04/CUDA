**Answer**

`cudaMalloc` **does not** return a pointer.  
It returns a `cudaError_t` value that indicates whether the allocation succeeded or failed. The pointer to device memory is supplied as an *output argument* (a pointer to a pointer).  

- **Success** – `cudaMalloc` returns `cudaSuccess` and the pointer argument is set to a non‑`NULL` value that points to the allocated memory on the device.  
- **Failure** – `cudaMalloc` returns a non‑`cudaSuccess` error code (e.g. `cudaErrorMemoryAllocation`, `cudaErrorInvalidValue`, etc.). In this case the pointer argument is **not** set to a new value; it remains unchanged (often still `NULL` if you initialized it that way before the call). It does **not** return a `nullptr` as a separate success code.

---

### Documentation

From the official CUDA C Programming Guide (section “Device Memory Allocation”):

> `cudaError_t cudaMalloc ( void **devPtr, size_t size );`  
>  Allocates `size` bytes of linear memory on the device and returns the
>  address of the allocated memory in `devPtr`.  
>  On success, returns `cudaSuccess`.  
>  On failure, returns an error code and **`devPtr` is not modified**.

---

### Test

```c
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int *devPtr = NULL;
    size_t hugeSize = (size_t) 1 << 60;  // ~1 EB, definitely too big

    cudaError_t err = cudaMalloc((void**)&devPtr, hugeSize);
    printf("cudaMalloc returned: %s\n", cudaGetErrorString(err));

    if (err != cudaSuccess) {
        printf("Pointer after failure is still: %p\n", (void*)devPtr);
    } else {
        printf("Allocated pointer: %p\n", (void*)devPtr);
    }

    return 0;
}
```

Compile and run:

```bash
nvcc test.cu -o test && ./test
```

Typical output:

```
cudaMalloc returned: out of memory
Pointer after failure is still: (nil)
```

The error code is not `cudaSuccess`, and the pointer remains unchanged (here it is still `NULL`). The program never receives a `nullptr` as a separate success code.

---

**Bottom line:**  
`cudaMalloc` returns an error code directly. The device pointer argument is only set when the call succeeds. If the allocation fails, the error code indicates the failure and the pointer remains whatever it was before the call.