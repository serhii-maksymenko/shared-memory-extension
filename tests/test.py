import time
import logging
import numpy as np
import multiprocessing
import shared_memory_extension

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('shared_memory_extension')
shared_memory_extension.set_log_level(3)

shared_memory_name = "test_memory"
image_shape = (480, 640, 3)
segment_size = np.prod(image_shape)
num_segments = 1
image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
logger.info(f'Created image of {image.nbytes} bytes')


def write(input_queue):
    shared_memory = shared_memory_extension.SharedMemory(segment_size, num_segments, shared_memory_name)
    while True:
        segment = shared_memory.get_free_segment()
        while segment is None:
            logger.info('No free segments, waiting...')
            time.sleep(0.05)
            segment = shared_memory.get_free_segment()
        segment.occupy()
        segment_id = segment.get_segment_id()
        segment.write(image)
        logger.info(f'{image.nbytes} bytes written to segment {segment_id}, shape: {image.shape}, dytpe: {image.dtype}')
        input_queue.put({
            'segment_id': segment_id
        })
        break


def read(input_queue):
    shared_memory = shared_memory_extension.SharedMemory(segment_size, num_segments, shared_memory_name)
    while True:
        input_data = input_queue.get()
        segment_id = input_data['segment_id']
        segment = shared_memory.get_segment(segment_id)
        image_from_segment = segment.read().reshape(image_shape)
        logger.info(f'{image_from_segment.nbytes} bytes read from segment {segment_id}, shape: {image_from_segment.shape}, dtype: {image_from_segment.dtype}')
        if np.array_equal(image, image_from_segment):
            logger.info('Test passed.')
        else:
            logger.error('Test failed.')
        segment.release()
        break


input_queue = multiprocessing.Queue()


write_process = multiprocessing.Process(target=write, args=(input_queue,))
read_process = multiprocessing.Process(target=read, args=(input_queue,))

write_process.start()
read_process.start()

write_process.join()
read_process.join()
