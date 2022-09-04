import os
from functools import wraps
from time import time

import numpy as np
from PIL import Image
from numba import jit


filepaths = [
    os.path.join(os.getcwd(), "frames", f)
    for f in os.listdir("frames")
    if f.endswith(".jpg")
]
images = [Image.open(fpath) for fpath in filepaths]
frames = np.array([np.asarray(im) for im in images])

MAX_ITERS = 1_00
MAX_DEQUE_LEN = 10
N_FRAMES = len(frames)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        t1 = time()
        result = f(*args, **kwargs)
        t2 = time()
        print(f"Function {f.__name__} executed in {(t2-t1):.4f}s")
        return result

    return wrap


@timeit
def numpy_blend_frames():
    blend_frames_q = np.empty([10, 1080, 1920, 3], dtype=np.uint8)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES]
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(
            blend_frames_q[1 : MAX_DEQUE_LEN - 1], [frame], axis=0
        )
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            blended_frame = np.average(blend_frames_q, axis=0).astype(np.uint8)


@timeit
def numpy_blend_frames_loop(x=np.zeros([10, 1080, 1920, 3], dtype=np.uint8)):
    blend_frames_q = np.empty_like(x)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES].reshape((1, 1080, 1920, 3))
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(blend_frames_q[1 : MAX_DEQUE_LEN - 1], frame, axis=0)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            total_frame = np.empty_like(frame[0])
            for frame in blend_frames_q:
                total_frame += frame
            blended_frame = (total_frame / blend_frames_q.shape[0]).astype(np.uint8)


@timeit
@jit(nopython=True)
def numba_blend_frames_nopython(x=np.zeros([10, 1080, 1920, 3], dtype=np.uint8)):
    blend_frames_q = np.empty_like(x)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES].reshape((1, 1080, 1920, 3))
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(blend_frames_q[1 : MAX_DEQUE_LEN - 1], frame, axis=0)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            total_frame = np.empty_like(frame[0])
            for frame in blend_frames_q:
                total_frame += frame
            blended_frame = (total_frame / blend_frames_q.shape[0]).astype(np.uint8)

@timeit
@jit(fastmath=True)
def numba_blend_frames_fastmath(x=np.zeros([10, 1080, 1920, 3], dtype=np.uint8)):
    blend_frames_q = np.empty_like(x)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES].reshape((1, 1080, 1920, 3))
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(blend_frames_q[1 : MAX_DEQUE_LEN - 1], frame, axis=0)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            total_frame = np.empty_like(frame[0])
            for frame in blend_frames_q:
                total_frame += frame
            blended_frame = (total_frame / blend_frames_q.shape[0]).astype(np.uint8)

@timeit
@jit(parallel=True)
def numba_blend_frames_parallel(x=np.zeros([10, 1080, 1920, 3], dtype=np.uint8)):
    blend_frames_q = np.empty_like(x)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES].reshape((1, 1080, 1920, 3))
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(blend_frames_q[1 : MAX_DEQUE_LEN - 1], frame, axis=0)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            total_frame = np.empty_like(frame[0])
            for frame in blend_frames_q:
                total_frame += frame
            blended_frame = (total_frame / blend_frames_q.shape[0]).astype(np.uint8)

@timeit
@jit(parallel=True, fastmath=True)
def numba_blend_frames_parallel_fastmath(x=np.zeros([10, 1080, 1920, 3], dtype=np.uint8)):
    blend_frames_q = np.empty_like(x)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES].reshape((1, 1080, 1920, 3))
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q = np.append(blend_frames_q[1 : MAX_DEQUE_LEN - 1], frame, axis=0)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            total_frame = np.empty_like(frame[0])
            for frame in blend_frames_q:
                total_frame += frame
            blended_frame = (total_frame / blend_frames_q.shape[0]).astype(np.uint8)


if __name__ == "__main__":
    numpy_blend_frames()
    numpy_blend_frames_loop()
    numba_blend_frames_nopython()
    numba_blend_frames_fastmath()
    numba_blend_frames_parallel()
    numba_blend_frames_parallel_fastmath()
