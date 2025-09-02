```cpp
/*
Aim of the program:
Use `thrust::unique` to count the number of unique elements in a sorted vector.

Thinking:
The task is to demonstrate how to use the Thrust library’s `unique` algorithm to
determine how many distinct elements are present in a sorted sequence stored on
the GPU.  
We will:

1. Include the necessary headers (`<thrust/unique.h>`, `<thrust/device_vector.h>`,
   `<thrust/host_vector.h>`, `<iostream>`).
2. Create a sorted `thrust::device_vector<int>` containing duplicate values.
3. Apply `thrust::unique` to the vector, which will reorder the vector so that
   the unique elements are at the beginning and return an iterator pointing
   just past the last unique element.
4. Calculate the number of unique elements by subtracting the original
   beginning iterator from the returned iterator.
5. Print the result.

The program is self‑contained, uses only Thrust (which runs on CUDA),
and can be compiled with `nvcc`. It will print the count of unique
elements to standard output.

*/

#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main()
{
    // Example sorted vector with duplicates
    // Sorted order: 1, 2, 2, 3, 3, 3, 4
    thrust::device_vector<int> d_vec(7);
    d_vec[0] = 1;
    d_vec[1] = 2;
    d_vec[2] = 2;
    d_vec[3] = 3;
    d_vec[4] = 3;
    d_vec[5] = 3;
    d_vec[6] = 4;

    // Apply thrust::unique to remove consecutive duplicates
    auto new_end = thrust::unique(d_vec.begin(), d_vec.end());

    // Number of unique elements
    size_t unique_count = new_end - d_vec.begin();

    std::cout << "Number of unique elements: " << unique_count << std::endl;

    // Optional: print the unique elements
    thrust::host_vector<int> h_vec(d_vec.begin(), new_end);
    std::cout << "Unique elements: ";
    for (int val : h_vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```