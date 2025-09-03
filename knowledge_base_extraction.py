#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <filesystem>

// Helper macro to wrap CUDA calls for error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << cudaGetErrorString(result) << "\" for " << func << std::endl;
        exit(99);
    }
}

// A simple kernel that simulates some ongoing work on the GPU.
// In a real application, this would be a complex computation.
__global__ void workload_kernel(float *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simulate a computation step
        data[idx] = sinf(data[idx]) * cosf(data[idx]) + 0.1f;
    }
}

// Structure to hold the state of our "process".
// This includes all the data that needs to be migrated between GPUs.
struct GpuProcessState {
    float* d_data = nullptr; // Pointer to the data on the current GPU
    size_t data_size = 0;
    int current_gpu = -1;
};

/**
 * @brief Initializes the process state and allocates memory on a specific GPU.
 * @param state The GpuProcessState object to initialize.
 * @param gpu_id The ID of the GPU to start the process on.
 * @param num_elements The number of float elements to process.
 */
void initialize_process(GpuProcessState& state, int gpu_id, size_t num_elements) {
    state.current_gpu = gpu_id;
    state.data_size = num_elements * sizeof(float);

    std::cout << "--> Initializing process on GPU " << gpu_id << std::endl;
    
    // Set the active device for the current host thread
    checkCudaErrors(cudaSetDevice(gpu_id));

    // Allocate memory on the selected GPU
    checkCudaErrors(cudaMalloc(&state.d_data, state.data_size));

    // Create some initial host data
    std::vector<float> h_data(num_elements);
    for(size_t i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Copy initial data from host to the GPU
    checkCudaErrors(cudaMemcpy(state.d_data, h_data.data(), state.data_size, cudaMemcpyHostToDevice));
    std::cout << "    Process initialized successfully.\n" << std::endl;
}

/**
 * @brief Simulates running a batch of work on the currently active GPU.
 * @param state The current state of the process.
 * @param iterations The number of kernel launches to perform.
 */
void run_workload(GpuProcessState& state, int iterations) {
    std::cout << "--> Running workload on GPU " << state.current_gpu << " for " << iterations << " iterations..." << std::endl;
    
    // Ensure the correct device is set for this thread
    checkCudaErrors(cudaSetDevice(state.current_gpu));

    int threads_per_block = 256;
    int blocks_per_grid = (state.data_size / sizeof(float) + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < iterations; ++i) {
        workload_kernel<<<blocks_per_grid, threads_per_block>>>(state.d_data, state.data_size / sizeof(float));
        // In a real app, you might check for a migration signal here
    }
    
    // Wait for all work on the current GPU to complete before proceeding
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "    Workload completed.\n" << std::endl;
}


/**
 * @brief Migrates the entire process state from its current GPU to a new one.
 * @param state The process state to migrate.
 * @param target_gpu_id The ID of the GPU to move the process to.
 */
void migrate_process(GpuProcessState& state, int target_gpu_id) {
    if (state.current_gpu == target_gpu_id) {
        std::cout << "--> Already on target GPU " << target_gpu_id << ". No migration needed." << std::endl;
        return;
    }

    std::cout << "=====================================================" << std::endl;
    std::cout << ">>> Starting Migration from GPU " << state.current_gpu << " to GPU " << target_gpu_id << " <<<" << std::endl;
    std::cout << "=====================================================\n" << std::endl;
    
    // 1. PAUSE: Synchronize to ensure all pending work on the source GPU is finished.
    // This is our "pause" step.
    std::cout << "    Step 1: Pausing and synchronizing work on source GPU " << state.current_gpu << std::endl;
    checkCudaErrors(cudaSetDevice(state.current_gpu));
    checkCudaErrors(cudaDeviceSynchronize());

    // 2. SAVE STATE: Allocate temporary host memory to hold the GPU data.
    std::vector<float> h_temp_data(state.data_size / sizeof(float));
    
    // 3. COPY TO HOST: Copy the data from the source GPU device to host memory.
    std::cout << "    Step 2: Copying data from source GPU " << state.current_gpu << " to Host RAM." << std::endl;
    checkCudaErrors(cudaMemcpy(h_temp_data.data(), state.d_data, state.data_size, cudaMemcpyDeviceToHost));
    
    // 4. CLEANUP SOURCE: Free the memory on the source GPU.
    std::cout << "    Step 3: Freeing resources on source GPU " << state.current_gpu << std::endl;
    checkCudaErrors(cudaFree(state.d_data));
    state.d_data = nullptr;

    // 5. SET NEW TARGET: Change the active device for the current thread to the target GPU.
    std::cout << "    Step 4: Setting active device to target GPU " << target_gpu_id << std::endl;
    checkCudaErrors(cudaSetDevice(target_gpu_id));
    state.current_gpu = target_gpu_id;

    // 6. ALLOCATE ON TARGET: Allocate memory on the new target GPU.
    checkCudaErrors(cudaMalloc(&state.d_data, state.data_size));

    // 7. COPY TO TARGET: Copy the saved state from host memory to the new GPU.
    std::cout << "    Step 5: Copying data from Host RAM to target GPU " << target_gpu_id << std::endl;
    checkCudaErrors(cudaMemcpy(state.d_data, h_temp_data.data(), state.data_size, cudaMemcpyHostToDevice));

    // 8. RESUME: The migration is complete. The process can now resume its work on the new GPU.
    std::cout << "\n>>> Migration Complete! Process is now running on GPU " << target_gpu_id << " <<<\n" << std::endl;
}

/**
 * @brief Copies the final data from the GPU to the host and saves it to a text file.
 * @param state The current state of the process.
 * @param filename The name of the file to save the output to.
 */
void save_output_to_file(GpuProcessState& state, const std::string& filename) {
    std::cout << "--> Saving final output to file: " << filename << std::endl;

    // Ensure the correct device is active
    checkCudaErrors(cudaSetDevice(state.current_gpu));

    // Create a host vector to store the results
    size_t num_elements = state.data_size / sizeof(float);
    std::vector<float> h_results(num_elements);

    // Copy the final data from the GPU device back to the host
    std::cout << "    Copying " << num_elements << " elements from GPU " << state.current_gpu << " to host..." << std::endl;
    checkCudaErrors(cudaMemcpy(h_results.data(), state.d_data, state.data_size, cudaMemcpyDeviceToHost));

    // Open the output file
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the data to the file, one number per line for readability
    std::cout << "    Writing data to " << filename << "..." << std::endl;
    for (size_t i = 0; i < num_elements; ++i) {
        outfile << h_results[i] << "\n";
    }

    outfile.close();
    std::cout << "    Successfully saved output.\n" << std::endl;
}


/**
 * @brief Cleans up and frees all allocated resources.
 * @param state The process state to clean up.
 */
void cleanup_process(GpuProcessState& state) {
    if (state.d_data != nullptr) {
        std::cout << "--> Cleaning up resources on GPU " << state.current_gpu << std::endl;
        checkCudaErrors(cudaSetDevice(state.current_gpu));
        checkCudaErrors(cudaFree(state.d_data));
        state.d_data = nullptr;
        std::cout << "    Cleanup complete." << std::endl;
    }
}


int main() {
    int device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "This program requires at least 2 GPUs to demonstrate migration." << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA-capable GPUs.\n" << std::endl;

    // Define our process and its workload size
    GpuProcessState my_process;
    size_t num_elements = 1024 * 1024 * 10; // 10 million floats

    // =======================================================
    // PHASE 1: Start on GPU 0 and do some work
    // =======================================================
    int initial_gpu = 0;
    initialize_process(my_process, initial_gpu, num_elements);
    run_workload(my_process, 100); // Run 100 iterations

    std::cout << "Process is running. Simulating a 3-second pause before migration..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));


    // =======================================================
    // PHASE 2: Migrate the process from GPU 0 to GPU 1
    // =======================================================
    int target_gpu = 1;
    migrate_process(my_process, target_gpu);


    // =======================================================
    // PHASE 3: Continue working on GPU 1
    // =======================================================
    run_workload(my_process, 150); // Run 150 more iterations on the new GPU


    // =======================================================
    // PHASE 4: Save the final output
    // =======================================================
    std::string output_dir = "output";
    // Note: Creating a directory requires the C++17 standard or later.
    // Ensure your host compiler (like g++) supports it.
    // You may need to add `-std=c++17` to your nvcc command line.
    if (!std::filesystem::exists(output_dir)) {
        std::cout << "--> Creating output directory: " << output_dir << std::endl;
        std::filesystem::create_directory(output_dir);
    }
    save_output_to_file(my_process, output_dir + "/gpu_process_migrator.txt");


    // =======================================================
    // PHASE 5: Cleanup
    // =======================================================
    cleanup_process(my_process);

    return 0;
}

