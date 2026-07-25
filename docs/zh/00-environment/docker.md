# Docker 使用

部分昇腾平台提供预装 CANN 的容器镜像。使用容器时，需要将 NPU 设备、驱动目录和数据目录挂载到容器中。

先检查 Docker：

```bash
docker --version
```

具体镜像名称、设备节点和挂载目录由平台提供。进入容器后，继续运行本章的两项检查：

```bash
npu-smi info
python -c 'import torch, torch_npu; print(torch.npu.is_available())'
```

平台直接提供已配置好的终端时，可以在宿主环境完成课程，不需要额外创建容器。
