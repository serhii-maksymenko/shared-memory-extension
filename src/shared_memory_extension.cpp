#include <Python.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "spdlog/spdlog.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;


void set_log_level(int level) {
    spdlog::set_level((spdlog::level::level_enum)level);
}


typedef struct SharedMemorySegment {
    PyObject_HEAD
public:
    py::size_t segment_size_;
    int segment_id_;
    void *segment_memory_ = NULL;
    void *segment_data_memory_ = NULL;

    SharedMemorySegment(int segment_size, int segment_id, void *shared_memory) : segment_size_(segment_size), segment_id_(segment_id) {
        segment_memory_ = (void*)((uint8_t*)shared_memory + ((segment_size + 1) * segment_id));
        segment_data_memory_ = (void*)((uint8_t*)segment_memory_ + 1);
        spdlog::info("Create shared memory segment {} of size {} bytes", segment_id_, segment_size_);
    }

    void write(py::array_t<uint8_t> image_array) {
        py::size_t size = image_array.size();
        if (size != segment_size_) {
            throw std::runtime_error("Shared Memory size and input data size don't match");
        }

        spdlog::debug("Write {} bytes to segment {}", size, segment_id_);
        
        memcpy(segment_data_memory_, image_array.request().ptr, size);
    }

    py::array_t<uint8_t> read() {
        // Create a NumPy array from the shared memory data
        auto result = py::array_t<uint8_t>(segment_size_, (uint8_t*)segment_data_memory_);

        return result;
    }

    int get_segment_id() {
        return segment_id_;
    }

    void release() {
        bool status = false;
        void* status_data = &status;
        memcpy(segment_memory_, status_data, 1);
    }

    bool is_occupied() {
        bool status = false;
        void* status_data = &status;
        memcpy(status_data, segment_memory_, 1);

        return status;
    }

    void occupy() {
        bool status = true;
        void* status_data = &status;
        memcpy(segment_memory_, status_data, 1);
    }

    ~SharedMemorySegment() {
        spdlog::info("Destroy shared memory segment {}", segment_id_);
    }
} SharedMemorySegment;


typedef struct SharedMemory {
    PyObject_HEAD
public:
    int shm_fd_ = -1;
    void *shared_memory_ = NULL;
    int size_;
    int segment_size_;
    int num_segments_;
    const char* name_ = NULL;
    std::vector<SharedMemorySegment*> segments;

    SharedMemory(int segment_size, int num_segments, const char* name) {
        // We need to copy string which comes from Python because otherwise it will be released before destroy() call
        char *temp_name = (char*)malloc(strlen(name));
        strcpy(temp_name, name);
        name_ = temp_name;
        segment_size_ = segment_size;
        num_segments_ = num_segments;
        size_ = (segment_size + 1) * num_segments;
        shm_fd_ = shm_open(name, O_CREAT | O_RDWR, 0666);
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to open shared memory object");
        }

        // Set the size of the shared memory object
        if (ftruncate(shm_fd_, size_) == -1) {
            throw std::runtime_error("Failed to set shared memory object size");
        }

        // Map the shared memory into the process's address space
        shared_memory_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (shared_memory_ == MAP_FAILED) {
            throw std::runtime_error("Failed to map shared memory");
        }

        for (int i = 0; i < num_segments; i++) {
            auto segment = new SharedMemorySegment(segment_size_, i, shared_memory_);
            segments.emplace_back(segment);
        }

        spdlog::info("Create `{}` shared memory of size {} bytes", name_, size_);
    }

    ~SharedMemory() {
        if (shared_memory_ != NULL) {
            munmap(shared_memory_, size_);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }

        for (int i = 0; i < num_segments_; i++) {
            delete segments.at(i);
        }

        spdlog::info("Destroy `{}` shared memory", name_);

        if (name_ != NULL) {
            shm_unlink(name_);
            free((void*)name_);
        }
    }

    SharedMemorySegment* get_free_segment() {
        for (int i = 0; i < num_segments_; i++) {
            auto segment = segments.at(i);
            if (!segment->is_occupied()) {
                return segment;
            }
        }

        return NULL;
    }

    SharedMemorySegment* get_segment(int segment_idx) {
        return segments.at(segment_idx);
    }
} SharedMemory;


PYBIND11_MODULE(shared_memory_extension, m) {
    m.def("set_log_level", &set_log_level);
    py::class_<SharedMemory>(m, "SharedMemory")
        .def(py::init<int, int, const char*>())
        .def("get_free_segment", &SharedMemory::get_free_segment, py::return_value_policy::reference)
        .def("get_segment", &SharedMemory::get_segment, py::return_value_policy::reference);

    py::class_<SharedMemorySegment>(m, "SharedMemorySegment")
        .def(py::init<int, int, void*>())
        .def("write", &SharedMemorySegment::write)
        .def("read", &SharedMemorySegment::read)
        .def("release", &SharedMemorySegment::release)
        .def("is_occupied", &SharedMemorySegment::is_occupied)
        .def("occupy", &SharedMemorySegment::occupy)
        .def("get_segment_id", &SharedMemorySegment::get_segment_id);
}
