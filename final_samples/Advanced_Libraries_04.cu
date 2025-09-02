```cpp
/*
Use Thrust to sort a large array of key-value pairs. Create one `device_vector` for keys and another for values. Use `thrust::sort_by_key` to sort both vectors based on the key values.

Thinking:
The goal is to demonstrate how to use Thrust to sort keys and keep values in sync with the sorted keys. The program will:
1. Define a large number of key-value pairs (e.g., 10 million).
2. Allocate two `thrust::device_vector`s, one for keys and one for values.
3. Initialize the host vectors with random data. For reproducibility, the keys can be random integers and values can be the original indices or another random set.
4. Copy the host vectors to the device vectors.
5. Use `thrust::sort_by_key(keys.begin(), keys.end(), values.begin());` to sort the keys and simultaneously reorder the values accordingly.
6. Copy the sorted vectors back to host and optionally print a few entries to verify that sorting worked.
7. Measure execution time to show performance benefits.

The code will be a self‑contained .cu file that can be compiled with `nvcc`. It uses standard Thrust containers and algorithms, demonstrates memory transfer, and provides a small validation output.

*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <iostream>
#include <iomanip>
#include <chrono>

int main()
{
    const std::size_t N = 10000000; // 10 million elements

    // Host vectors
    thrust::host_vector<int> h_keys(N);
    thrust::host_vector<int> h_values(N);

    // Fill keys with random integers and values with original indices
    thrust::default_random_engine rng(42);
    thrust::uniform_int_distribution<int> dist(0, static_cast<int>(N));

    for (std::size_t i = 0; i < N; ++i)
    {
        h_keys[i]   = dist(rng);
        h_values[i] = static_cast<int>(i); // keep track of original index
    }

    // Device vectors
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;

    std::cout << "Sorting " << N << " key-value pairs using thrust::sort_by_key..." << std::endl;

    // Timing
    auto start = std::chrono::high_resolution_clock::now();

    // Sort by key; values are reordered accordingly
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sorting completed in " << elapsed.count() << " seconds." << std::endl;

    // Copy back to host for validation
    thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

    // Validate: first 10 sorted keys and corresponding values
    std::cout << std::setw(12) << "Index" << std::setw(12) << "Key" << std::setw(12) << "OriginalIdx" << std::endl;
    for (std::size_t i = 0; i < 10 && i < N; ++i)
    {
        std::cout << std::setw(12) << i
                  << std::setw(12) << h_keys[i]
                  << std::setw(12) << h_values[i] << std::endl;
    }

    return 0;
}
```