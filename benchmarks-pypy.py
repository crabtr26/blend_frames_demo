import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from time import time

import numpy as np
from PIL import Image

import rust_ext


filepaths = [
    os.path.join(os.getcwd(), "frames", f)
    for f in os.listdir("frames")
    if f.endswith(".jpg")
]
images = [Image.open(fpath) for fpath in filepaths]
frames = np.array([np.asarray(im) for im in images])
frames_dq = deque(maxlen=10)
for frame in frames:
    frames_dq.append(frame)

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


def blend_frame_algorithm(frames: deque):
    nframes = len(frames)
    avgnp = np.zeros(frames[0].shape)
    for frame in frames:
        try:
            avgnp += frame
        except:
            pass
    avgnp = avgnp / nframes
    avgnpi = np.rint(avgnp).astype(np.uint8)
    return avgnpi


@timeit
def benchmark_numpy_average():
    np.average(frames, axis=0).astype(np.uint8)

@timeit
def benchmark_numpy_average_loop():
    total_frame = np.empty_like(frames[0])
    for frame in frames:
        total_frame += frame
    blended_frame = (total_frame / frames.shape[0]).astype(np.uint8)


@timeit
def benchmark_blend_frame_algorithm():
    blend_frame_algorithm(frames_dq)


@timeit
def benchmark_rust_average():
    rust_ext.average(frames)


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
def deque_blend_frames():
    blend_frames_q = deque(maxlen=MAX_DEQUE_LEN)
    input_frame_idx = 0
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES]
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q.append(frame)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            blended_frame = blend_frame_algorithm(blend_frames_q)


@timeit
def deque_blend_frames_multithreading():
    blend_frames_q = deque(maxlen=MAX_DEQUE_LEN)
    input_frame_idx = 0
    futures = []
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES]
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q.append(frame)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(blend_frame_algorithm, blend_frames_q)
                futures.append(future)

    results = [future.result() for future in futures]


@timeit
def deque_blend_frames_multiprocessing():
    blend_frames_q = deque(maxlen=MAX_DEQUE_LEN)
    input_frame_idx = 0
    futures = []
    for _ in range(MAX_ITERS):
        frame = frames[input_frame_idx % N_FRAMES]
        if frame is None:
            break
        input_frame_idx += 1
        blend_frames_q.append(frame)
        if input_frame_idx % MAX_DEQUE_LEN == 0:
            with ProcessPoolExecutor() as executor:
                future = executor.submit(blend_frame_algorithm, blend_frames_q)
                futures.append(future)

    results = [future.result() for future in futures]


@timeit
def rust_blend_frames():
    rust_ext.blend_frames(frames, MAX_ITERS, MAX_DEQUE_LEN)


if __name__ == "__main__":
    benchmark_numpy_average()
    benchmark_numpy_average_loop()
    benchmark_blend_frame_algorithm()
    benchmark_rust_average()
    print("\n")

    numpy_blend_frames()
    numpy_blend_frames_loop()

    deque_blend_frames()
    deque_blend_frames_multithreading()
    deque_blend_frames_multiprocessing()

    rust_blend_frames()
