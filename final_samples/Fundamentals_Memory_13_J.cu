When `cudaMalloc()` fails, the pointer passed as its first argument **does not get modified**; it keeps whatever value it had before the call.

**Why?**

* `cudaMalloc()` is defined to set the value of the pointer only when the allocation succeeds.  
* Internally the function creates a temporary pointer, performs the allocation, and if the allocation is successful it copies that temporary pointer into the location pointed to by the user’s argument.  
* If the allocation fails, the temporary pointer remains uninitialized, and the function simply returns the error code without touching the user‑supplied pointer.

Therefore, after a failed `cudaMalloc()` the pointer remains unchanged (which is usually `NULL` or whatever it was set to earlier). Always check the return status before using the pointer.