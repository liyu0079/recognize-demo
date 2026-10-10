# 多模态视觉检测全链路优化

## 已落地的架构

```text
上传媒体
  -> 预处理（一次解码、一次 RGB）
  -> 检测层
       阶段 1: 通用短提示词组
       阶段 2: 根据阶段 1 标签按需补检
       阶段 3: 触发式 2x2 / 完整模式 3x3 放大切片
       -> 坐标回映射 -> Soft-NMS -> raw_candidates
  -> 分割层: 高价值前 MAX_SEGMENTED_OBJECTS 个候选 -> SAM2
  -> 信息层: 整图 OCR、姿态/手部姿态、延迟 Caption
  -> 压缩掩码、摘要、诊断和性能指标
```

调用关系：`main.py` 负责 API、模式、视频和输出契约；`openvino_engine.py` 只做模型推理、编译缓存和 SAM2；`recall_scheduler.py` 是无模型依赖的阶段调度、切片和 Soft-NMS 策略；`full_capabilities.py` 负责 OCR/姿态/Caption 的真实性与输出规范。

## 调度与后处理

`fast` 是默认模式：跑全图阶段 1，阶段 2 按首轮标签扩展；当候选覆盖率低、没有目标或小目标密集时才执行 2x2 切片。`full` 强制执行 3x3 切片，仍不把 SAM2 从 24 个直接提升到 80 个。通过 `DEFAULT_RUN_MODE`、`ENABLE_STAGED_RECALL`、`ENABLE_TILE_RECALL` 可灰度启停，旧 PyTorch 回退路径仍可用。

伪代码：

```python
base = detect(image, BASE_PROMPTS)
detail_prompts = prompts_for_labels(base.labels)
detail = detect(image, detail_prompts)
if full_mode or sparse_or_dense_small_objects(base + detail):
    tiles = [detect(upscale(tile), detail_prompts or SMALL_OBJECT_PROMPTS) for tile in grid]
    candidates = remap_to_original(base + detail + tiles)
final = conservative_semantic_deduplicate(adaptive_soft_nms(candidates))
```

Soft-NMS 对同类大目标使用较高 IoU 阈值，对小目标使用较低阈值并衰减而非删除低分框。颜色只提供最高 12% 的排序权重。父子类仅在 IoU 大于 0.82 时去重；以前会把番茄与水果类在 0.10 IoU 就删除的规则不再进入主路径。响应中保留 `raw_candidates`，可直接判断是模型未产出候选还是后处理后排名落选。

为兼顾“应检尽检”和开放词检测的语义稳定性，`license plate`、`manhole cover`、`air conditioner`、`tomato` 使用类别专属验证而非普通 NMS：车牌必须被车辆框承载；井盖必须是下半部的低饱和近方形小区域；空调外机必须满足小尺寸、比例和灰白机身约束；番茄必须存在近圆形红/橙/黄果实色块。验证失败的框仍会保留在 `raw_candidates`，不会作为最终物体或进入 SAM2。

## 模块变更、收益与回滚

| 优化 | 预期收益 | 风险 | 回滚 |
| --- | --- | --- | --- |
| 三阶段提示词 | 细分类召回提升，避免十组长词同时编码 | 场景错判会多一次检测 | `ENABLE_STAGED_RECALL=0` |
| 触发式切片 | 远景车辆、番茄、车牌、井盖等小目标召回提升 | full 模式时延增加 | `ENABLE_TILE_RECALL=0` 或 `run_mode=fast` |
| 自适应 Soft-NMS | 密集同类不再被硬删 | 低分重复框可能增加 | 仅在 `recall_scheduler.py` 恢复旧 NMS |
| 三层检测/分割 | SAM2 故障或预算不足不丢 bbox | 低优先级框无 mask | 调整 `MAX_SEGMENTED_OBJECTS`，不建议直接设 80 |
| OCR 独立对象 | 文字最小框可追溯，避免每框重复 OCR | OCR 冷启动影响首张 | 关闭 OCR 模型或保留空文字字段 |
| Caption fail-closed | 不把原始权重伪装成可用描述 | IR 未导出时无 Caption | 导出 Stateful IR 后按需开启 |
| 视频场景帧和轨迹 | 少跑重复重推理，人员/车辆跨帧有 `track_id` | 镜头快速运动会断轨 | 调高场景阈值或只用固定帧 |
| OV 缓存/预热 | 降低编译和首帧抖动 | 个别插件不支持缓存配置 | `OPENVINO_WARMUP=0`，缓存失败自动降级 |

## Caption 启用门槛

`/api/health` 只有发现完整 Moondream2 Stateful OpenVINO IR 时才报告 `enrichment.caption.available=true`。缺 IR 时 API 返回空 `caption`，并明确说明不可用；不会使用 CPU safetensors 生成貌似稳定的结果。启用后也只应面向无 OCR 的高价值框或用户选中框异步投递，不对所有检测框批量生成。

已提供两个按需接口：`POST /api/segment-region` 接收图片和 `bbox` JSON 数组，仅为这一框运行 SAM2；`POST /api/caption-tasks` 接收图片和 `bbox`，返回任务 ID，再由 `GET /api/caption-tasks/{task_id}` 轮询。Caption IR 缺失时创建接口返回 `unavailable`，不会进入队列或生成占位文本。

## 视频策略

视频先保留基础固定采样，再使用缩小灰度直方图扫描场景变化，把变化明显帧补入关键帧集合，最多 `VIDEO_MAX_SAMPLE_COUNT` 帧。关键帧使用完整调度，非关键帧后续可扩展为轻量检测；当前 API 已将车辆和人员以标签 IoU 关联为 `track_id`，摘要只采用代表帧，避免跨帧重复计数。

## 回归测试集与指标模板

测试集目录建议采用如下布局，标注文件使用 COCO 或每图 JSON，包含 bbox、类别、OCR 文本和可选 mask：

```text
tests/vision-regression/
  dense-tomatoes/
  distant-vehicles/
  plates-and-chinese-signs/
  small-air-conditioners/
  manholes-and-road-furniture/
  multi-person-hands/
  small-object-street/
```

每轮固定输出下表，并保留同一阈值、同一设备、至少三次热身后的 p50/p95：

| 版本 | 场景/类别 | GT | TP | FP | 召回率 | 误检率 | 单图目标数 | 小目标召回率 | OCR 框准确率 | mask 覆盖率 | 平均 ms | p95 ms | CPU 峰值 | 内存峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | dense-tomatoes/tomato | | | | | | | | | | | | | |
| staged-fast | dense-tomatoes/tomato | | | | | | | | | | | | | |
| staged-full | dense-tomatoes/tomato | | | | | | | | | | | | | |

可用 `backend/evaluate_regression.py` 直接请求本地 API 并写出 CSV：

```powershell
backend\.venv\Scripts\python.exe backend\evaluate_regression.py tests\vision-regression\manifest.json --run-mode full --output regression-full.csv
backend\.venv\Scripts\python.exe backend\evaluate_regression.py tests\vision-regression\manifest.json --run-mode fast --output regression-fast.csv
```

`manifest.json` 为 `{"cases":[{"id":"dense-tomatoes","image":"dense-tomatoes/a.jpg","objects":[{"label":"tomato","bbox":[x1,y1,x2,y2]}],"ocr_texts":["文字"],"mask_required":["tomato"]}]}`。评估器会输出每图 TP、FP、召回率、误检率、OCR 准确率、mask 覆盖率、DINO/SAM2/OCR/总耗时与进程资源峰值。

接口结果的 `timings_ms` 已包含 `preprocess`、`dino_detection`、`nms_postprocess`、`sam2_segmentation`、`ocr`、`caption` 和 `total`。`resource_usage` 返回进程 CPU 时间与 tracemalloc 峰值；系统级 CPU RSS/GPU 指标可在基准运行器中额外采集。

## 建议排期

1. 第 1 周：用上述七类图像建立带标注基线；分别灰度 `fast` 和 `full`，先确认召回与重复框变化。
2. 第 2 周：按类别调阶段二词表、Soft-NMS 阈值和切片触发阈值；只接受召回提升且误检率受控的变更。
3. 第 3 周：已补齐异步 Caption 查询接口和用户选框分割接口；下一步评估 OCR PP-OCRv5/PP-StructureV3。
4. 第 4 周：导出并验证 Moondream2 Stateful IR；完成前保持 Caption 不可用状态。
5. 第 5 周：仅在回归集仍有稳定漏检时，评估 YOLO-World 或 OWLv2 作为第二召回器，不能替代当前策略基线。
