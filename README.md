# Shared Memory Extension for Python

Read/write NumPy arrays with dtype==np.uint8 (aka images) into a shared memory for inter-process communication. It can be applied for image or video processing tasks.

## Usage example

Write:
```python
import numpy as np
import shared_memory_extension


image_shape = (480, 640, 3)
segment_size = np.prod(image_shape)
num_segments = 1
image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
shared_memory = shared_memory_extension.SharedMemory(segment_size, num_segments, 'test_memory')
 # get_free_segment can return None if all segments are occupied
segment = shared_memory.get_free_segment()
segment.occupy()
segment_id = segment.get_segment_id()
segment.write(image)
```

Read:
```python
import shared_memory_extension


segment_id = ... # pass segment_id into another process
shared_memory = shared_memory_extension.SharedMemory(segment_size, num_segments, 'test_memory')
segment = shared_memory.get_segment(segment_id)
image_from_segment = segment.read().reshape(image_shape)
segment.release()
```

## Dependencies

C++:
- [spdlog](https://github.com/gabime/spdlog)
- [pybind11](https://github.com/pybind/pybind11)

Follow their installation instructions.

Python:
- setuptools
- numpy

## Build and install

Build:
```bash
python setup.py bdist_wheel
```

Install:
```bash
pip install dist/*
```

Quick test
```bash
python tests/test.py
```
