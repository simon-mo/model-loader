import time
import struct

import requests
import torch

from model_loader import download_to_device

def get_safe_tensor_metadata(url):
    # resolve any redirect
    resp = requests.head(url, allow_redirects=True)

    resolved_url = resp.url
    content_size = resp.headers.get("Content-Length")

    # Fetch the first 8 bytes of the file
    headers = {'Range': 'bytes=0-7'}
    response = requests.get(resolved_url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {'Range': f'bytes=8-{7 + length_of_header}'}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    tensor_header = response.json()

    return {
        "resolved_url": resolved_url,
        "content_size": content_size,
        "header_size": length_of_header + 8,
        "data_size": int(content_size) -length_of_header - 8,
        "tensor_header": tensor_header,
    }

class LoaderByteTensor:
    def __init__(self, ptr, nbytes):
        self.ptr = ptr
        self.nbytes = nbytes

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self.ptr, False), # second item is read-only flag
            "shape": (self.nbytes,),
            "typestr": "|u1",
        }

class SafeTensorLoader:
    def __init__(self, url):
        self.metadata = get_safe_tensor_metadata(url)
        print(self.metadata)
        self.raw_pointer = download_to_device(self.metadata["resolved_url"], self.metadata["header_size"], self.metadata["data_size"])
        self.data = self._create_torch_tensors(self.metadata, self.raw_pointer)

    def _create_torch_tensors(self, metadata, raw_pointer):
        data = {}
        for key, value in metadata["tensor_header"].items():
            if key == "__metadata__":
                continue

            dtype = value["dtype"]
            shape = value["shape"]
            start_offset, end_offset = value["data_offsets"]

            torch_dtype = {
                "BF16": torch.bfloat16,
                "F16": torch.float16,
                "F32": torch.float32,
            }[dtype]

            # create torch tensor from pointer
            ptr = raw_pointer + start_offset
            our_tensor = LoaderByteTensor(ptr, nbytes=end_offset-start_offset)
            untyped_storage = torch.as_tensor(our_tensor, device=torch.device("cuda")).untyped_storage()
            torch_tensor = torch.tensor(untyped_storage, device=torch.device("cuda"), dtype=torch_dtype).view(shape)

            assert torch_tensor.numel() == (end_offset-start_offset) // torch_tensor.element_size()
            assert torch_tensor.data_ptr() == ptr

            data[key] = torch_tensor

        return data

    def keys(self):
        return self.data.keys()

    def get_tensor(self, key):
        return self.data[key]


if __name__ == "__main__":
    url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6/resolve/main/model.safetensors"

    loader = SafeTensorLoader(url)
    print(loader.keys())

    import safetensors
    with safetensors.safe_open("model.safetensors", "pt", device="cuda") as f:
        all_tensor_names = set(loader.keys()) - {"__metadata__"}
        assert set(f.keys()) == set(loader.keys()), f"Keys don't match: {f.keys()} vs {loader.keys()}, missing {all_tensor_names - set(f.keys())}"

        for key in all_tensor_names:
            assert torch.allclose(f.get_tensor(key), loader.get_tensor(key))
            print(f"{key} OK")


