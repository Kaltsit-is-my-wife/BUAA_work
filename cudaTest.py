import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("GPU 个数:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.current_device())
    print("GPU 名称:", torch.cuda.get_device_name(0))

import torch, torchtext
print("torch =", torch.version)
print("torchtext =", torchtext.__version__)
