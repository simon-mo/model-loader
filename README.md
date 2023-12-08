Stream Models from Huggingface Model Hub Directly to GPU

The top level interface is
```
state_dict = model_loader.load(http_url, device="cuda")
```

Currently it supports both SafeTensors and PyTorch checkpoint formats.
Currently it is about 10x faster than Huggingface version.

Follow up:
vLLM integration
Optionally cache to disk?
Better interface for huggingface compatbility (non trivial)
