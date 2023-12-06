use std::ffi::c_void;

use cudarc::driver::sys as cu;
use indicatif::ProgressBar;
use pyo3::prelude::*;
use tokio_stream::StreamExt;

fn cuda_init() {
    unsafe {
        let result = cu::cuInit(0);
        assert!(
            result == cu::cudaError_enum::CUDA_SUCCESS,
            "init failed: {:?}",
            result
        );

        let mut device: cu::CUdevice = 0;
        let device_result = cu::cuDeviceGet(&mut device, 0);
        assert!(
            device_result == cu::cudaError_enum::CUDA_SUCCESS,
            "device get failed: {:?}",
            device_result
        );

        let mut context: cu::CUcontext = std::ptr::null_mut();
        let context_result = cu::cuCtxCreate_v2(&mut context, 0, device);
        assert!(
            context_result == cu::cudaError_enum::CUDA_SUCCESS,
            "context create failed: {:?}",
            context_result
        );
    }
}

fn cuda_malloc_pair(size: usize) -> (*mut c_void, u64) {
    unsafe {
        let mut ptr = std::ptr::null_mut();
        let alloc_result = cu::cuMemHostAlloc(
            &mut ptr,
            size,
            cu::CU_MEMHOSTALLOC_PORTABLE | cu::CU_MEMHOSTALLOC_WRITECOMBINED,
        );
        assert!(
            alloc_result == cu::cudaError_enum::CUDA_SUCCESS,
            "host alloc failed: {:?}",
            alloc_result
        );

        // alloc device memory
        let mut device_ptr = cu::CUdeviceptr::default();
        let device_alloc_result = cu::cuMemAlloc_v2(&mut device_ptr, size);
        assert!(
            device_alloc_result == cu::cudaError_enum::CUDA_SUCCESS,
            "device alloc failed: {:?}",
            device_alloc_result
        );

        (ptr, device_ptr)
    }
}

#[tokio::main]
async fn download_url(
    url: &str,
    header_size: usize,
    mut host_ptr: *mut c_void,
) -> Result<(), reqwest::Error> {
    let range_header = format!("bytes={}-", header_size);
    let resp = reqwest::Client::new()
        .get(url)
        .header("Range", range_header)
        .send()
        .await?;

    let bar = ProgressBar::new(resp.content_length().unwrap());

    let mut stream = resp.bytes_stream();
    while let Some(item) = stream.next().await {
        let bytes = item?;

        bar.inc(bytes.len() as u64);

        // write to the ptr
        unsafe {
            std::ptr::copy(bytes.as_ptr(), host_ptr as *mut u8, bytes.len());
        }

        // advance the ptr
        host_ptr = unsafe {
            let host_ptr = host_ptr as *mut u8;
            let host_ptr = host_ptr.add(bytes.len());
            host_ptr as *mut c_void
        };
    }

    Ok(())
}

#[pyfunction]
fn download_to_device(url: &str, header_size: usize, data_size: usize) -> PyResult<usize> {
    cuda_init();

    let (host_ptr, device_ptr) = cuda_malloc_pair(data_size);

    download_url(url, header_size, host_ptr).unwrap();

    // copy from host to device
    unsafe {
        let copy_result = cu::cuMemcpyHtoD_v2(device_ptr, host_ptr as *mut c_void, data_size);
        assert!(
            copy_result == cu::cudaError_enum::CUDA_SUCCESS,
            "copy failed: {:?}",
            copy_result
        );
    }

    Ok(device_ptr as usize)
}

/// A Python module implemented in Rust.
#[pymodule]
fn model_loader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download_to_device, m)?)?;
    Ok(())
}
