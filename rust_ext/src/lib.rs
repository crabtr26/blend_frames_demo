use bounded_vec_deque::BoundedVecDeque;
use numpy::ndarray::{Array, Array3, Array4, ArrayView3, ArrayView4, Axis};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray4};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use std::convert::TryFrom;

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn average(x: ArrayView4<u8>) -> Array3<u8> {
        x.mean_axis(Axis(0)).unwrap()
    }

    fn average_deque(deque: BoundedVecDeque<ArrayView3<u8>>) -> Array3<u8> {
        let n_frames_u8: u8 = 10;
        let mut total_frame = Array3::zeros((1080, 1920, 3));
        let iter = deque.iter();
        for frame in iter {
            total_frame += frame
        }
        return total_frame / n_frames_u8;
    }

    fn blend_frames(frames: ArrayView4<u8>, max_iters: usize) {
        let n_frames = frames.len();
        let mut deque = BoundedVecDeque::new(10);
        for i in 0..max_iters {
            let frame = frames.index_axis(Axis(0), i % 10);
            deque.push_back(frame);
            if i % 10 == 0 {
                let blended_frame = average_deque(deque.to_owned());
            }
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "average")]
    fn average_py<'py>(py: Python<'py>, x: PyReadonlyArray4<'_, u8>) -> &'py PyArray3<u8> {
        average(x.as_array()).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "blend_frames")]
    fn blend_frames_py<'py>(py: Python<'py>, frames: PyReadonlyArray4<'_, u8>, max_iters: usize) {
        blend_frames(frames.as_array(), max_iters)
    }

    Ok(())
}
