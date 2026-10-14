#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr int kSkip = 77;

std::string csv_escape(const std::string& text) {
    std::string escaped = "\"";
    for (const char value : text) {
        if (value == '\"') escaped += '\"';
        escaped += value;
    }
    return escaped + "\"";
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

bool append_record(const char* path,
                   const cudaDeviceProp& device,
                   const char* compiler_version,
                   const int runtime_version,
                   const int driver_version,
                   const std::size_t count,
                   const std::size_t requested_bytes,
                   const std::size_t total_bytes,
                   const std::size_t free_before,
                   const std::size_t free_after_alloc,
                   const std::size_t free_after_release,
                   const char* allocation_status) {
    const std::filesystem::path output_path(path);
    std::error_code directory_error;
    std::filesystem::create_directories(output_path.parent_path(), directory_error);
    if (directory_error) return false;
    std::ofstream output(output_path, std::ios::app);
    if (!output) return false;
    if (output.tellp() == 0) {
        output << "timestamp_utc,backend,device,input_elements,dtype,compiler,compiler_version,"
                  "runtime_version,driver_version,"
                  "requested_bytes,device_memory_total_bytes,device_memory_free_before_bytes,"
                  "device_memory_free_after_alloc_bytes,device_memory_free_after_release_bytes,"
                  "allocation_status\n";
    }
    output << utc_timestamp() << ",cuda," << csv_escape(device.name) << ',' << count
           << ",float32,nvcc," << compiler_version << ','
           << runtime_version << ',' << driver_version << ','
           << requested_bytes << ',' << total_bytes << ',' << free_before << ','
           << free_after_alloc << ',' << free_after_release << ','
           << allocation_status << '\n';
    return output.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: add_cuda_memory_eval <results.csv> <positive_N>\n";
        return 2;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(static_cast<std::size_t>(-1) / (3 * sizeof(float)))) {
        std::cerr << "N must be a positive integer whose three float32 buffers fit size_t.\n";
        return 2;
    }
    const std::size_t count = static_cast<std::size_t>(parsed);
    const std::size_t bytes_per_buffer = count * sizeof(float);
    const std::size_t requested_bytes = 3 * bytes_per_buffer;

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        status == cudaErrorInitializationError || (status == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: CUDA device/driver unavailable\n";
        return kSkip;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    int device_id = 0;
    cudaDeviceProp device{};
    int runtime_version = 0;
    int driver_version = 0;
    status = cudaGetDevice(&device_id);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&device, device_id);
    if (status == cudaSuccess) status = cudaRuntimeGetVersion(&runtime_version);
    if (status == cudaSuccess) status = cudaDriverGetVersion(&driver_version);
    if (status != cudaSuccess) {
        std::cerr << "CUDA environment query failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    status = cudaMemGetInfo(&free_before, &total_bytes);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemGetInfo before allocation failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    float* buffers[3] = {nullptr, nullptr, nullptr};
    const char* allocation_status = "allocated";
    for (float*& buffer : buffers) {
        status = cudaMalloc(reinterpret_cast<void**>(&buffer), bytes_per_buffer);
        if (status != cudaSuccess) {
            allocation_status = cudaGetErrorString(status);
            break;
        }
    }
    std::size_t free_after_alloc = 0;
    std::size_t ignored_total = 0;
    if (cudaMemGetInfo(&free_after_alloc, &ignored_total) != cudaSuccess) free_after_alloc = 0;
    for (float* buffer : buffers) {
        if (buffer != nullptr) cudaFree(buffer);
    }
    std::size_t free_after_release = 0;
    if (cudaMemGetInfo(&free_after_release, &ignored_total) != cudaSuccess) free_after_release = 0;

    if (!append_record(argv[1], device, ADD_CUDA_COMPILER_VERSION, runtime_version, driver_version, count, requested_bytes,
                       total_bytes, free_before, free_after_alloc, free_after_release, allocation_status)) {
        std::cerr << "Could not append memory evaluation CSV: " << argv[1] << '\n';
        return 1;
    }
    std::cout << "CUDA memory evaluation: N=" << count
              << " requested_bytes=" << requested_bytes
              << " allocation_status=" << allocation_status << '\n';
    return status == cudaSuccess ? 0 : 1;
}
