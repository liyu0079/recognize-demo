# 本地 OpenVINO 视觉标注

FastAPI + Vue 3 本地视觉标注系统。运行时保留 Grounding DINO、SAM2-small、RTMPose-tiny 三个已随仓库提供的 OpenVINO 能力，并为 RTMPose-hand、PaddleOCR、Moondream2 提供真实本地 IR 的懒加载接入口。默认使用 Intel Arc 的 OpenVINO `GPU.0`，失败自动回退 CPU；缺失权重时字段保持为空并在 `/api/health` 标明，不生成伪结果。

## 快速开始

```powershell
powershell -ExecutionPolicy Bypass -File .\backend\setup-backend.ps1
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1
pnpm install
pnpm dev
```

首次准备在可联网机器完成下载与 IR 导出；之后复制 `backend\models` 即可离线部署。浏览器前端默认地址为 `http://127.0.0.1:5173`，健康检查为 `http://127.0.0.1:8000/api/health`。

## 引擎切换

```powershell
# 默认：OpenVINO，优先 GPU.0
$env:USE_OPENVINO="1"

# 兼容兜底：保留的 PyTorch CPU 路径
$env:USE_OPENVINO="0"

# 审计完整六能力模型包
backend\.venv\Scripts\python.exe backend\export_all_full_models.py --check-only
```

接口保留对象原字段 `label`、`score`、`bbox`、`mask`，只追加 `keypoints`。前端支持 COCO 17 点骨架显示/隐藏，坐标与框、掩码均以原图像素为准。

部署细节、Arc 驱动校验、显存与坐标问题排查见 [backend/OPENVINO_DEPLOY.md](backend/OPENVINO_DEPLOY.md)。
