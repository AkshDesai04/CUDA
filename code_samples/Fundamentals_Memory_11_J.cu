If you forget to copy a value into a __constant__ variable before launching a kernel, the device simply reads whatever happens to be stored in that constant‑memory slot.  
In practice, CUDA guarantees that all device memory—including the constant address space—is **zero‑initialized** when the device starts (or when a context is created). Therefore, the value seen by the kernel will be 0 (or a sequence of zero bytes).  

In other words:

* **What happens?** The kernel executes with the constant memory still holding the default zero values.  
* **What is the initial value?** Zero for all bits (0x00…00).  

Keep in mind that relying on this implicit zero‑initialization is fragile; if you ever change the memory allocation method (e.g., using dynamic constant memory, or if the device is reused across streams without resetting) you should explicitly initialize the constant memory with `cudaMemcpyToSymbol()` or by defining an initial value in the declaration (`__constant__ int c = 5;`).