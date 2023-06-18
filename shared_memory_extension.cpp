#include <Python.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <numpy/arrayobject.h>
#include "spdlog/spdlog.h"


typedef struct SharedMemory {
    PyObject_HEAD
public:
    int shm_fd_ = -1;
    void *shared_memory_ = NULL;
    int size_;
    const char* name_ = NULL;

    int create(int size, const char* name) {
        // we need to copy string which comes from Python because otherwise it will be released before destroy() call
        char *temp_name = (char*)malloc(strlen(name));
        strcpy(temp_name, name);
        name_ = temp_name;
        size_ = size;
        shm_fd_ = shm_open(name, O_CREAT | O_RDWR, 0666);
        if (shm_fd_ == -1) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to open shared memory object");

            return 1;
        }

        // Set the size of the shared memory object
        if (ftruncate(shm_fd_, size) == -1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set shared memory object size");

            return 1;
        }

        // Map the shared memory into the process's address space
        shared_memory_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (shared_memory_ == MAP_FAILED)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to map shared memory");

            return 1;
        }

        spdlog::info("Create `{}` shared memory of size {} bytes", name_, size_);

        return 0;
    }

    int destroy() {
        if (shared_memory_ != NULL) {
            munmap(shared_memory_, size_);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }

        spdlog::info("Destroy `{}` shared memory", name_);

        if (name_ != NULL) {
            shm_unlink(name_);
            free((void*)name_);
        }

        return 0;
    }

    int write(PyArrayObject* image_array) {
        npy_intp size = PyArray_SIZE(image_array) * PyArray_ITEMSIZE(image_array);
        if (size != size_) {
            PyErr_SetString(PyExc_RuntimeError, "Shared Memory size and input data size don't match");
            return 1;
        }

        spdlog::info("Write {} bytes to `{}` shared memory", size, name_);
        
        memcpy(shared_memory_, PyArray_DATA(image_array), size);

        return 0;
    }

    PyObject* read() {
        // Create a NumPy array from the shared memory data
        npy_intp dims[] = { size_ };
        PyObject* array_obj = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, shared_memory_);
        if (array_obj == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
            return NULL;
        }

        // Create a copy of the NumPy array to ensure data ownership
        PyObject* array_copy = PyArray_Copy(reinterpret_cast<PyArrayObject*>(array_obj));
        if (array_copy == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array copy");
            Py_DECREF(array_obj);
            return NULL;
        }

        // Clean up
        Py_DECREF(array_obj);

        return array_copy;
    }
} SharedMemory;


static int SharedMemory_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    int size;
    const char *name;
    if (!PyArg_ParseTuple(args, "is", &size, &name)) {
        return 1;
    }

    ((SharedMemory *) self)->create(size, name);
    return 0;
}

// Define the destructor function for the class
static void SharedMemory_dealloc(SharedMemory* self) {
    self->destroy();
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* SharedMemory_write(SharedMemory* self, PyObject* args) {
    PyObject* image_obj;
    if (!PyArg_ParseTuple(args, "O", &image_obj))
        return Py_BuildValue("i", 1);
    
    if (!PyArray_Check(image_obj)) {
        return Py_BuildValue("i", 1);
    }

    PyArrayObject* image_array = (PyArrayObject*)PyArray_FROM_OTF(image_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (image_array == NULL) {
        return Py_BuildValue("i", 1);
    }

    int status = self->write(image_array);

    Py_DECREF(image_array);

    return Py_BuildValue("i", status);
}

static PyObject* SharedMemory_read(SharedMemory* self) {
    spdlog::info("SharedMemory_read");
    return self->read();
}

// Define the class type object
static PyMethodDef SharedMemory_methods[] = {
    {"write", (PyCFunction)SharedMemory_write, METH_VARARGS, "write data"},
    {"read", (PyCFunction)SharedMemory_read, METH_NOARGS, "read data"},
    {nullptr, nullptr, 0, nullptr}  // Sentinel
};

static PyTypeObject SharedMemoryType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "SharedMemory",
    .tp_basicsize = sizeof(SharedMemory),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) SharedMemory_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
    .tp_doc = PyDoc_STR("SharedMemory"),
    .tp_methods = SharedMemory_methods,
    // .tp_members = heapctype_members,
    .tp_init = (initproc) SharedMemory_init,
    .tp_new = PyType_GenericNew,
};

// Module initialization function
static PyModuleDef shared_memory_extension_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "shared_memory_extension",
    .m_doc = "SharedMemory API",
    .m_size = -1
};

// Initialize the module
PyMODINIT_FUNC PyInit_shared_memory_extension(void) {
    import_array();
    // Add the class to the module
    if (PyType_Ready(&SharedMemoryType) < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&shared_memory_extension_module);
    if (!module) {
        return nullptr;
    }

    
    Py_INCREF(&SharedMemoryType);
    if (PyModule_AddObject(module, "SharedMemory", reinterpret_cast<PyObject*>(&SharedMemoryType)) < 0) {
        Py_DECREF(&SharedMemoryType);
        Py_DECREF(module);
    }

    return module;
}
