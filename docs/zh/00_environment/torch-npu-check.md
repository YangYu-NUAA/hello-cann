# PyTorch + torch_npu 最小校验

## 本节目标

确认当前 Python 环境能够使用 `torch_npu` 调用昇腾 NPU。

## 前置条件

- 已安装 CANN。
- 已安装 PyTorch 和 `torch_npu`。
- 已加载 CANN 环境变量。

如果还不知道环境脚本在哪里，可以先查：

```bash
find /usr/local/Ascend ~/Ascend -name set_env.sh 2>/dev/null
```

有些容器里 `/usr/local/Ascend` 只挂载了 `driver`。这种情况下继续找 toolkit 的实际安装路径，或者先用下面的导入测试判断当前 Python 环境是否已经可用。

## 校验脚本

```bash
python - <<'PY'
import torch
import torch_npu

x = torch.ones((2, 3), device="npu")
y = x + 1

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("device:", y.device)
print(y.cpu())
PY
```

## 期望结果

- 脚本没有报错。
- 输出中能看到 `npu` 设备。
- `y.cpu()` 的结果为全 2 张量。
