# OpenVINO Arc 部署

## 首次准备

1. 安装 Intel Arc Windows 驱动，并在 PowerShell 执行：

```powershell
cd D:\codeList\recognize-demo
powershell -ExecutionPolicy Bypass -File .\backend\setup-backend.ps1
```

2. 脚本在首次运行下载官方权重、导出 `backend\models\openvino\*.xml/.bin`。导出成功后，后端推理不下载任何内容，断网可运行。
3. 验证 Arc OpenVINO 插件：

```powershell
.\backend\.venv\Scripts\python.exe -c "from openvino import Core; print(Core().available_devices)"
```

输出包含 `GPU.0` 或 `GPU` 时，服务会优先使用 Arc 130T；不同 OpenVINO 版本对单卡的命名不同，缺失或某个模型编译失败时自动回退 `CPU`。

## 启动与切换

```powershell
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1
pnpm dev
```

访问 `http://127.0.0.1:8000/api/health`。`engine=openvino`、`device=GPU` 或 `GPU.0` 表示加速已启用；检测、SAM2 和姿态模型首次真正使用时才编译加载。

切换到保留的 PyTorch CPU 兜底：

```powershell
$env:USE_OPENVINO="0"
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1
```

恢复默认 OpenVINO：

```powershell
Remove-Item Env:USE_OPENVINO
```

## 故障排查

| 现象 | 处理 |
| --- | --- |
| health 显示缺少 IR | 在已联网的准备机运行 `python backend/export_all_models.py`，将整个 `backend/models` 拷贝到离线机器。 |
| health 为 CPU | 更新 Intel Arc 驱动；用 `Core().available_devices` 确认 `GPU` 或 `GPU.0`。CPU 是设计内的容错回退。 |
| 显存不足或卡顿 | 将 `MAX_IMAGE_LONG_EDGE` 设为 `1024` 或 `896`；减少 prompt 类别；`OPENVINO_IDLE_SECONDS=30` 可更快释放闲置编译图。 |
| SAM2 导出失败 | 确认使用仓库内 `sam2_hiera_s.yaml` 与 small 权重；执行 `python export_all_models.py --clean`，不要混用 base/large 权重。 |
| 框、掩码、骨架偏移 | 检查导出与运行均使用 1024 x 1024 SAM2 输入、192 x 256 RTMPose 输入；接口返回值始终由 `OpenVINOCompatibleManager` 反缩放到原图。 |
| 姿态为空 | 只有精确标签 `person` 执行 RTMPose，且每人必须有 17 个有效 SimCC 点；检查 `rtmpose_tiny.xml/.bin`。 |

## 性能原则

- 原始媒体长边统一限制为 1280，结果 bbox、mask、keypoints 均映射回原始绝对像素。
- SAM2 图片编码每帧仅运行一次，之后逐框使用固定单框 decoder，避免动态 shape 触发 GPU 重新编译。
- 请求串行执行，避免检测、分割、姿态并发占用 Arc 显存。模型闲置达到 `OPENVINO_IDLE_SECONDS` 后卸载，下次请求透明重载。
