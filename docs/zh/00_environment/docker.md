# Docker 使用

先确认当前机器是否提供 Docker：

```bash
docker --version
```

本次华为云实验环境返回 `docker: command not found`，因此第一版实验直接使用云平台预置的宿主机环境。Docker 不是 00 章的通过条件。

后续在具备 Docker 的机器上补充镜像选择、NPU 设备挂载、数据目录和权限设置。容器启动后仍使用 [torch-npu-check.md](torch-npu-check.md) 作为验收脚本。
