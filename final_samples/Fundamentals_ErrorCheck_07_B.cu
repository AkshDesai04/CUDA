```cuda
/*
Aim: Provide a list of 3 common asynchronous CUDA API calls (`cudaMemcpyAsync`, kernel launch, etc.).

Thinking: 
The user requested a list of three asynchronous CUDA API calls. 
To satisfy this request while following the developer’s higher‑priority instruction to output a .cu file, 
I’ll create a minimal CUDA C program that prints the desired list. 
The program will compile with nvcc and run on any CUDA‑enabled device. 
It will include the typical asynchronous calls in the comment: 
1. cudaMemcpyAsync 
2. Kernel launch 
3. cudaStreamSynchronize (or cudaEventRecord/WaitFor).
*/

#include <stdio.h>

int main(void)
{
    printf("1. cudaMemcpyAsync\n");
    printf("2. Kernel launch (e.g., myKernel<<<grid, block>>> )\n");
    printf("3. cudaStreamSynchronize (or cudaEventRecord/WaitFor)\n");
    return 0;
}
```