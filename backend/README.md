# OpenVINO 本地视觉后端

后端在本机运行 Grounding DINO、SAM2-small、RTMPose-tiny。媒体和提示词只进入本地 FastAPI；运行期不会下载权重或调用外部服务。

首次在可联网的准备机执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-backend.ps1
```

该脚本安装 CPU PyTorch 与 OpenVINO，下载官方权重并产生 `models\openvino` 下的 IR。部署到离线机器时复制整个 `backend\models` 目录。

启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1
```

默认 `USE_OPENVINO=1`，优先使用 `backend/models/openvino/` 中的三模型 IR；需要故障兜底时设置 `$env:USE_OPENVINO="0"`，严格走项目内保留的 PyTorch CPU 路径。

核心配置位于 `main.py`：`BOX_CONFIDENCE_THRESHOLD` 默认 `0.3`、`MAX_IMAGE_LONG_EDGE` 默认 `1280`、`OPENVINO_IDLE_SECONDS` 默认 `90`。完整驱动验证、性能调优和故障排查见 [OPENVINO_DEPLOY.md](OPENVINO_DEPLOY.md)。

`POST /api/analyze-media` 的字段保持原接口兼容。对象在原有 `label`、`score`、`bbox`、`mask` 基础上新增 `keypoints`；只有 `person` 返回 COCO 17 点数组，其他类别返回空数组。
