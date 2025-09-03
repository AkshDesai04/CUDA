```cpp
/*
Aim: Write a generic `device_copy` function that takes two `void*` pointers, a size, and calls `cudaMemcpy` with the DtoD kind.

Thinking:
1. The function should be generic and operate on raw pointers.  
2. It should accept a destination pointer, a source pointer, and the number of bytes to copy.  
3. The appropriate `cudaMemcpy` flag for device-to-device copies is `cudaMemcpyDeviceToDevice`.  
4. Returning the CUDA error code (`cudaError_t`) gives the caller visibility into success/failure.  
5. Include the necessary headers (`cuda_runtime.h` for CUDA APIs and `<cstddef>` for `size_t`).  
6. Keep the implementation minimal: just a single return statement that forwards the result of `cudaMemcpy`.  
7. No additional functionality or example usage is required; the focus is solely on providing this utility function.
*/

#include <cuda_runtime.h>
#include <cstddef>

// Generic device-to-device copy function
cudaError_t device_copy(void* dst, const void* src, size_t size)
{
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}
```