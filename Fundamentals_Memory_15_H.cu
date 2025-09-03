```cpp
/*
Declare a `__device__` variable in a header file. Does this cause problems? (Yes, can lead to multiple definitions unless handled with `extern`).

Thinking:
- In CUDA, a `__device__` variable defined in a header file and included in multiple translation units will be compiled into each unit, resulting in multiple definitions during linking.
- To avoid this, the header should only declare the variable with `extern __device__`, while the actual definition should reside in a single .cu file.
- The following example demonstrates both the problematic approach and the correct one using `extern`.
- The header file `dev_var.h` is simulated inside this single .cu file for simplicity.
- Two separate .cu files (simulated here as separate sections) include the header. The first includes the definition (causing a compile error if both were compiled), the second only declares it with `extern`. This illustrates the issue and the solution.
- Since we cannot create separate files here, the example is illustrative rather than directly compilable as-is.
*/

#include <cstdio>
#include <cuda_runtime.h>

/* Simulated header file: dev_var.h */
#ifndef DEV_VAR_H
#define DEV_VAR_H

/* 
 * Problematic definition: if this header is included in multiple
 * .cu files, each will have its own definition, causing multiple
 * definitions at link time.
 */
#ifdef INCLUDE_DEVICE_VAR_DEFINITION
__device__ int devVar = 42;  // definition
#else
/* Correct declaration: use extern to refer to a single definition. */
extern __device__ int devVar;
#endif

#endif /* DEV_VAR_H */

/* 
 * First translation unit: defines the device variable.
 * In practice, this would be in a separate .cu file.
 */
#define INCLUDE_DEVICE_VAR_DEFINITION
#include "dev_var.h"
#undef INCLUDE_DEVICE_VAR_DEFINITION

/* 
 * Second translation unit: declares the device variable.
 * In practice, this would be in another separate .cu file.
 */
#include "dev_var.h"

/* Kernel that uses the device variable */
__global__ void useVarKernel()
{
    printf("devVar = %d\\n", devVar);
}

int main()
{
    /* Launch kernel */
    useVarKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    /* Check for errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```