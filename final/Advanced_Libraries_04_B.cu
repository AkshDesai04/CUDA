```cpp
/*
Aim: Use thrust::unique to count the number of unique elements in a sorted vector.

Thinking:
- The goal is to demonstrate how to use thrust::unique to remove consecutive duplicates
  from a sorted vector and count how many unique elements remain.
- thrust::unique removes duplicate values that are adjacent in the sequence, so the input
  must be sorted.  In practice we can sort the vector first to guarantee this property.
- The function returns an iterator pointing to the new logical end of the range after
  duplicates are removed.  The distance from the original begin iterator to this new
  end iterator gives the number of unique elements.
- For illustration we will create a device_vector<int> containing some repeated values,
  sort it, apply thrust::unique, then compute and display the count as well as the
  unique values themselves.
- All necessary headers are included and the program is fully self‑contained.
- Compile with: nvcc -arch=sm_61 unique_count.cu -o unique_count
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iostream>

int main()
{
    // Sample data with duplicates
    thrust::host_vector<int> h_data = {1, 2, 2, 3, 3, 3, 4, 5, 5, 6};

    // Transfer data to device
    thrust::device_vector<int> d_vec = h_data;

    // Ensure the vector is sorted
    thrust::sort(d_vec.begin(), d_vec.end());

    // Remove consecutive duplicates
    auto new_end = thrust::unique(d_vec.begin(), d_vec.end());

    // Compute number of unique elements
    size_t unique_count = new_end - d_vec.begin();

    // Copy unique elements back to host for display
    thrust::host_vector<int> h_unique(d_vec.begin(), new_end);

    // Output results
    std::cout << "Number of unique elements: " << unique_count << std::endl;
    std::cout << "Unique elements: ";
    for (size_t i = 0; i < h_unique.size(); ++i)
    {
        std::cout << h_unique[i];
        if (i + 1 < h_unique.size())
            std::cout << ", ";
    }
    std::cout << std::endl;

    return 0;
}
```