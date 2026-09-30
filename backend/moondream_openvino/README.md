# Moondream2 专用 OpenVINO 导出

这是面向 `vikhyatk/moondream2` 原生快照的定向导出器。它不调用
`optimum-intel`，也不会在运行期使用网络。视觉前缀、文本提示和后续生成 token
共享同一个 OpenVINO Stateful KV Cache。

## 环境

Python 3.10+、PyTorch 2.2+ 和 OpenVINO 2024.3+ 是最低要求。建议使用项目的
Python 3.11 虚拟环境，并安装本目录的依赖：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r .\moondream_openvino\requirements.txt
```

导出前必须已有完整的本地模型快照，其中包含 `model.safetensors`、`tokenizer.json`
及其原生 Python 文件。导出程序不会自动访问 HuggingFace：

```powershell
cd backend
.\.venv\Scripts\python.exe .\export_moondream2.py `
  --model ..\weights\moondream2 `
  --output .\models\openvino\moondream2 `
  --precision fp16 --max-context 2048 --device arc
```

输出包括：

- `vision_encoder.xml/.bin`：含 BGR→RGB、Resize(378)、归一化的 SigLIP 编码器。
- `vision_projector.xml/.bin`：将全局/局部 27x27 特征投影为 729 个视觉 token。
- `token_embedding.xml/.bin`：本地分词器 ID 到 Moondream 词嵌入。
- `decoder.xml/.bin`：24 层 Phi-2 变体，`past_key_values` 已经转换为 Stateful
  `ReadValue/Assign`。
- `decoder_config.json`、`tokenizer.json` 与 `moondream2_ov_config.json`。

视觉导出会在 CPU 上将 OpenVINO 和 PyTorch 输出做余弦相似度比较，低于 `0.999`
会中断导出并保留错误，而不会留下被标记为可用的损坏模型。

## 推理

```powershell
.\.venv\Scripts\python.exe .\example_moondream2_ov.py `
  --model .\models\openvino\moondream2 `
  --image C:\images\sample.jpg --device GPU.0

.\.venv\Scripts\python.exe .\example_moondream2_ov.py `
  --model .\models\openvino\moondream2 `
  --image C:\images\sample.jpg --prompt "What is in this image?" --device GPU.0
```

运行器先尝试 `GPU.0`，Arc 插件或编译失败时如实回退 CPU，并输出实际 `device`。
调用者只传图片和提示词；KV 缓存由单一 OpenVINO `InferRequest` 管理，`reset()`
会清空全部 24 层状态。

## 设计边界

`openvino_genai.LLMPipeline` 只能加载其已注册的文本/视觉架构，不能给未注册的
Moondream2 自定义模型注入视觉 prefix embedding。因此该实现使用相同的 OpenVINO
Stateful 图机制，而不是把不兼容的 `decoder.xml` 伪装成 GenAI 模型。它仍然完全使用
OpenVINO IR 与 Arc GPU，不使用 PyTorch 或在线服务进行推理。

INT8 参数被保留，但必须先提供本地校准数据和评估集。工具会拒绝未校准的 INT8
请求，防止 Caption 质量在不知情的情况下退化。

## 常见问题

- **`Moondream2 export failed` 或进程内存不足**：FP16 1.7B 权重在导出时通常需要
  至少 16 GB 可用系统内存。关闭后端、浏览器和其他模型进程后重试。
- **`GPU.0` 不存在**：更新 Intel Arc 驱动，并以 `benchmark_app -d GPU` 验证 OpenVINO
  GPU 插件。Windows 上推荐使用与 OpenVINO 2024.3 或 2025.0 匹配的最新 Arc 驱动。
- **Stateful 变换失败**：确认 `openvino>=2024.3`，不要混用 2023.x 的
  `openvino.runtime` 与新的 `openvino` 包。
- **视觉校验低于 0.999**：不要忽略。通常是模型快照与随附 Python 代码版本不一致，
  或导出期间发生了不受控量化。
