# Full capability deployment

## 1. 能力与私有化边界

推理进程不访问第三方在线 API，也不会把图片上传到外部服务。公开模型的
首次下载只发生在准备阶段，运行机默认只读 `backend/models`。若确实需要在
启动时补齐资产，显式设置 `$env:AUTO_DOWNLOAD_MODELS="1"`；不设置时启动
只做本地审计，不因可选模型缺失而退出。

The runtime remains local-only and keeps the `USE_OPENVINO=1/0` switch. The
three bundled IR groups are Grounding DINO, SAM2-small and RTMPose-tiny. The
additional groups are loaded only when real local assets are supplied:

- `rtmpose_hand.xml/.bin`: 21-point hand SimCC model.
- `paddleocr.xml/.bin` plus a local PaddleOCR runtime/model directory.
- `moondream2.xml/.bin` plus the matching OpenVINO GenAI tokenizer/runtime.

Run the audit after the domestic network download has completed:

```powershell
backend\.venv\Scripts\python.exe backend\export_all_full_models.py --check-only
```

On a preparation machine, export the existing three models and convert local
ONNX assets by setting `RTMPOSE_HAND_ONNX`, `PADDLEOCR_ONNX`, and
`MOONDREAM2_ONNX`, then run `export_all_full_models.py`. The command fails if a
required asset is missing; it never creates placeholder weights.

`GET /api/health` exposes `capabilities`, `enrichment`, and the local asset
audit. GroundingDINO uses the original English checkpoint; text/referring
prompts should therefore use English categories such as `person`, `car`, or
`traffic sign`.

Every object returns compact `mask` RLE plus `mask_size`, `keypoints`,
`hand_keypoints`, `ocr_text`, and `caption`. Missing model capabilities return
empty fields and a health warning, never generated text or synthetic points.

## 2. Python 3.11 环境修复

在 Windows PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\backend\setup-backend.ps1 -SkipModelExport
```

脚本会检查 `py -3.11` 或 `python` 是否为 3.11；发现 `.venv` 的
`pyvenv.cfg` 指向不存在的解释器时会重建。若两者都不存在，安装 Python 3.11
并重新执行脚本。依赖安装不包含 CUDA；OpenVINO GPU.0 失败时自动使用 CPU。

## 3. 模型下载与 MD5 审计

```powershell
backend\.venv\Scripts\python.exe backend\model_downloader.py
backend\.venv\Scripts\python.exe backend\export_all_full_models.py --check-only
```

`model_downloader.py` 使用 ModelScope 官方 `snapshot_download`（缓存与断点续传）、
控制台进度输出和 `MODEL_MD5_<NAME>` 环境变量；SDK 不可用时才尝试 Git LFS。
下面是下载器内置的六个官方 ModelScope 仓库：

| 资产 | 官方仓库 | 权重下载地址 | MD5 环境变量 |
| --- | --- | --- | --- |
| GroundingDINO | [AI-ModelScope/GroundingDINO](https://modelscope.cn/models/AI-ModelScope/GroundingDINO) | SDK snapshot | `MODEL_MD5_GROUNDING_DINO` |
| SAM2 | [AI-ModelScope/sam2-hiera-base-plus](https://modelscope.cn/models/AI-ModelScope/sam2-hiera-base-plus) | SDK snapshot | `MODEL_MD5_SAM2_SMALL` |
| RTMPose-hand | [litert-community/RTMPose-Hand-LiteRT](https://modelscope.cn/models/litert-community/RTMPose-Hand-LiteRT) | SDK snapshot | `MODEL_MD5_RTMPOSE_HAND` |
| RTMPose-s | [litert-community/RTMPose-s-LiteRT](https://modelscope.cn/models/litert-community/RTMPose-s-LiteRT) | SDK snapshot | `MODEL_MD5_RTMPOSE_TINY` |
| PaddleOCR-VL | [PaddlePaddle/PaddleOCR-VL](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL) | SDK snapshot | `MODEL_MD5_PADDLEOCR` |
| Moondream2 | [AI-ModelScope/moondream2](https://modelscope.cn/models/AI-ModelScope/moondream2) | SDK snapshot | `MODEL_MD5_MOONDREAM2` |

下载器通过 ModelScope SDK 访问上表仓库，不使用裸 HTTP 权重链接。未设置 MD5 时状态为 `unverified`，不会把本地首次计算值冒充官方
校验值；设置期望值后不匹配状态为 `failed`，对应模块会被跳过。

```powershell
$env:MODEL_MD5_GROUNDING_DINO = '<从官方清单填入 32 位小写 MD5>'
$env:MODEL_MD5_SAM2_SMALL = '<从官方清单填入 32 位小写 MD5>'
$env:MODEL_MD5_RTMPOSE_HAND = '<从官方清单填入 32 位小写 MD5>'
$env:MODEL_MD5_RTMPOSE_TINY = '<从官方清单填入 32 位小写 MD5>'
$env:MODEL_MD5_PADDLEOCR = '<从官方清单填入 32 位小写 MD5>'
$env:MODEL_MD5_MOONDREAM2 = '<从官方清单填入 32 位小写 MD5>'
$env:AUTO_DOWNLOAD_MODELS = '1'
```

### 3.1 本地权重梳理与重复文件合并

下载器的六项资产审计现在还会检查每个快照中的关键权重文件，而不是仅凭
README 或配置文件判定“存在”。当前 `weights/` 下的六个原始快照均已具备关键
文件；`checksum=unverified` 只表示没有提供官方期望 MD5，并不代表文件为空。

ModelScope 的缓存与目标目录可能各保留一份完全相同的二进制文件。需要明确
合并时运行：

```powershell
backend\.venv\Scripts\python.exe backend\model_downloader.py --no-download --dedupe
```

该命令只对当前六个仓库 ID 的快照做内容指纹比对，并把相同文件替换为硬链接；
两个路径都会保留，不能创建硬链接时会跳过且不删除源文件。旧的错误仓库缓存
（例如 `IDEA-Research--GroundingDINO`、`liter-community--RTMPose-Hand-LiteRT`）
只作陈旧标记，不会被自动删除。`backend/models/openvino` 中的 XML/BIN 是转换后
运行时资产，和 `weights/` 原始权重格式不同，也不会被合并或删除。

注意：原始权重齐全不等于当前服务的六项推理链全部就绪。当前运行目录仍需
分别提供 `rtmpose_hand.xml/.bin` 以及 Moondream2 的 OpenVINO GenAI IR；否则
健康检查会如实报告手部关键点或 Caption 不可用，并返回空字段。
另外，下载器为兼容旧接口保留了 `sam2_small`、`rtmpose_tiny` 键名，但当前仓库
实际分别是 `sam2-hiera-base-plus` 与 `RTMPose-s-LiteRT`；它们不是严格意义上的
SAM2-small/RTMPose-tiny，同名 IR 不能直接互换，需在导出时确认模型变体。

如果准备机需要切换仓库版本，可用对应 `MODELSCOPE_<NAME>_MODEL_ID` 环境变量覆盖
默认仓库 ID；下载器仍只通过 SDK 或 Git LFS 访问 ModelScope。仓库不可访问、文件缺失
或校验失败只会在健康接口标记异常，不会让 FastAPI 退出，也不会生成伪造 OCR、Caption
或关键点。

## 4. 导出与启动

```powershell
backend\.venv\Scripts\python.exe backend\export_all_full_models.py
powershell -ExecutionPolicy Bypass -File .\backend\start-backend.ps1
```

前端依赖修复与启动：

```powershell
corepack enable
corepack prepare pnpm@latest --activate
pnpm install
pnpm run dev -- --host 0.0.0.0 --port 5173
```

若系统只有 npm，则使用 `npm install` 与 `npm run dev -- --host 0.0.0.0`。

## 5. 分步验收

1. 环境：`backend\.venv\Scripts\python.exe --version`、`pnpm --version`。
2. 资产：按固定顺序执行以下命令，下载器会自动下载缺失文件、显示进度并校验 MD5：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\backend\setup-backend.ps1 --SkipModelExport
   .\backend\.venv\Scripts\python.exe .\backend\model_downloader.py
   .\backend\.venv\Scripts\python.exe .\backend\export_all_full_models.py --check-only
   powershell -ExecutionPolicy Bypass -File .\backend\start-backend.ps1
   powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1
   ```

   检查 `model_checksums.json` 与终端中的 `verified/unverified/failed/missing`。
3. 后端：访问 `http://127.0.0.1:8000/api/health`，确认 `model_assets`、
   `capabilities`、`enrichment` 与本地文件一致；`model_progress.percent`
   表示已通过完整性审计的资产比例。
4. 前端：打开 `http://127.0.0.1:5173`，上传图片，先验证框/掩码/17 点，
   再验证本地 OCR、Caption、手部 IR 启用后的悬浮信息。
5. 代码：`pnpm run typecheck`、`pnpm run build`，以及
   `python -m py_compile backend/*.py`。
