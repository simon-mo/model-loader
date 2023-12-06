use cudarc::driver::sys as cu;
use pyo3::prelude::*;

#[tokio::main]
async fn download_url(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let resp = reqwest::get(url).await?;
    let body = resp.bytes().await?;
    Ok(body.to_vec())
}

#[pyfunction]
fn fast_download(url: &str) -> PyResult<String> {
    Ok(download_url(url).unwrap().len().to_string())
}

#[pyfunction]
fn cuda_malloc(size: usize) -> PyResult<usize> {
    unsafe {
        let result = cu::cuInit(0);
        assert!(result == cu::cudaError_enum::CUDA_SUCCESS, "init failed: {:?}", result);

        let mut device: cu::CUdevice = 0;
        let device_result = cu::cuDeviceGet(&mut device, 0);
        assert!(device_result == cu::cudaError_enum::CUDA_SUCCESS, "device get failed: {:?}", device_result);

        let mut context: cu::CUcontext = std::ptr::null_mut();
        let context_result = cu::cuCtxCreate_v2(&mut context, 0, device);
        assert!(context_result == cu::cudaError_enum::CUDA_SUCCESS, "context create failed: {:?}", context_result);

        let mut ptr = std::ptr::null_mut();
        let alloc_result = cu::cuMemHostAlloc(&mut ptr, size, cu::CU_MEMHOSTALLOC_PORTABLE | cu::CU_MEMHOSTALLOC_WRITECOMBINED);
        assert!(alloc_result == cu::cudaError_enum::CUDA_SUCCESS, "alloc failed: {:?}", alloc_result);

        println!("Allocated {} bytes at {:p} in host", size, ptr);

        // write 1s to the ptr
        let ones = vec![1u8; size];
        std::ptr::copy(ones.as_ptr(), ptr as *mut u8, size);

        // alloc device memory
        let mut device_ptr = cu::CUdeviceptr::default();
        let device_alloc_result = cu::cuMemAlloc_v2(&mut device_ptr, size);
        assert!(device_alloc_result == cu::cudaError_enum::CUDA_SUCCESS, "device alloc failed: {:?}", device_alloc_result);

        // copy from host to device
        let copy_result = cu::cuMemcpyHtoD_v2(device_ptr, ptr, size);
        assert!(copy_result == cu::cudaError_enum::CUDA_SUCCESS, "copy failed: {:?}", copy_result);

        // return the device ptr
        println!("Copied {} bytes to {:?} in device", size, device_ptr);

        Ok(device_ptr as usize)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn model_loader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_download, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_malloc, m)?)?;
    Ok(())
}
