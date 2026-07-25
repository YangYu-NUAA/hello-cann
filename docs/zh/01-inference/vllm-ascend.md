# vLLM-Ascend 服务化部署

vLLM-Ascend 可以在昇腾 NPU 上启动 OpenAI 兼容接口。应用接入时通常需要服务地址、模型名称和 API key。

本节将包括：

1. 选择与 CANN、PyTorch、`torch_npu` 匹配的 vLLM-Ascend 版本。
2. 使用单张 NPU 启动 Qwen 服务。
3. 通过 `/v1/chat/completions` 发送请求。
4. 记录首 token 延迟、输出吞吐和峰值显存。

开始本节前应先完成 [Transformers 单卡推理](transformers-torch-npu.md)。服务化实验完成后，命令和版本组合会补充到本页。
