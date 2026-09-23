# 运行手册

项目采用本地 FastAPI 和 Vue 3。默认链路为：媒体图片或视频关键帧 → 长边 1280 预处理 → Grounding DINO → 置信度筛选 → person ROI 的 RTMPose-tiny → SAM2-small → 原图坐标 JSON。

## 部署

1. 安装 Intel Arc 130T 驱动、Python 3.11、Node 20+。
2. 在可联网的准备机运行 `backend\setup-backend.ps1`，它会安装 CPU PyTorch/OpenVINO、下载官方权重并导出固定形状 IR。
3. 离线机器复制 `backend\models`，运行 `backend\start-backend.ps1` 与 `pnpm dev`。
4. 用 `Invoke-RestMethod http://127.0.0.1:8000/api/health` 确认 `engine` 为 `openvino`；出现 `GPU.0` 即 Arc 加速已生效，出现 `CPU` 是安全降级。

运行时不会联网下载或请求第三方服务。详情请见 [../backend/OPENVINO_DEPLOY.md](../backend/OPENVINO_DEPLOY.md)。

## 使用与验证

上传图片或视频；视频保持原有关键帧抽样和前端同步显示。返回对象的 bbox、mask、keypoints 全部是原媒体绝对像素坐标。前端“姿态”开关只绘制标签为 `person` 且有 17 个关键点的对象。

OpenVINO 出现问题时设置 `$env:USE_OPENVINO="0"` 并重启服务，验证保留的 PyTorch CPU 路径；恢复加速时改为 `1` 或移除该环境变量。
