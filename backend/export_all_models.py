"""一次性准备 Grounding DINO、SAM2-small、RTMPose-tiny 的 OpenVINO IR。

首次执行需要网络下载官方权重；成功后只使用 backend/models 内的本地文件。
运行时 main.py 不下载模型，也不访问任何外部服务。
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import openvino as ov
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IR_DIR = MODEL_DIR / "openvino"
GROUNDING_REPO = "IDEA-Research/grounding-dino-tiny"
SAM2_REPO = "facebook/sam2.1-hiera-small"
SAM2_CHECKPOINT = "sam2.1_hiera_small.pt"
# OpenMMLab 目前以 SDK zip 发布 RTMPose-tiny；zip 内含静态 256x192 ONNX。
RTMPOSE_SDK_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip"


def ensure_file(repo_id: str, filename: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return target
    configured = os.getenv("HF_ENDPOINT", "").strip().rstrip("/")
    endpoints = tuple(dict.fromkeys(value for value in (configured, "https://hf-mirror.com") if value))
    errors: list[str] = []
    for endpoint in endpoints:
        for attempt in range(2):
            try:
                print(f"[download] {repo_id}/{filename} via {endpoint} (attempt {attempt + 1})", flush=True)
                return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=directory, endpoint=endpoint, etag_timeout=20))
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
                time.sleep(1)
    raise RuntimeError("SAM2-small 权重下载失败；请设置 HF_ENDPOINT 或手动放入 " + str(directory / filename) + "。\n" + "\n".join(errors[-3:]))


def save_ir(model: object, example_input: object, name: str, output_names: tuple[str, ...], force: bool = False) -> None:
    """先保存固定 shape ONNX，再转换为 IR；端口名供运行时严格校验。"""
    IR_DIR.mkdir(parents=True, exist_ok=True)
    xml = IR_DIR / f"{name}.xml"
    if not force and xml.exists() and xml.with_suffix(".bin").exists():
        print(f"[skip] {name}: {xml}")
        return
    try:
        print(f"[convert] {name}: OpenVINO PyTorch frontend", flush=True)
        ov_model = ov.convert_model(model, example_input=example_input)
    except Exception as direct_error:
        print(f"[fallback] {name}: direct conversion failed: {direct_error}", flush=True)
        onnx_path = IR_DIR / f"{name}.onnx"
        if not onnx_path.exists():
            export_kwargs = {"opset_version": 17, "do_constant_folding": True, "output_names": list(output_names)}
            if isinstance(example_input, dict):
                if name == "grounding_dino":
                    # GroundingDino forward order is pixel_values, input_ids,
                    # token_type_ids, attention_mask, pixel_mask. Exporting
                    # kwargs assigns names by position and silently swaps ports.
                    ordered_names = ("pixel_values", "input_ids", "token_type_ids", "attention_mask", "pixel_mask")
                    ordered_inputs = tuple(example_input[key] for key in ordered_names)
                    export_kwargs["input_names"] = list(ordered_names)
                    torch.onnx.export(model, ordered_inputs, f=str(onnx_path), **export_kwargs)
                else:
                    export_kwargs["input_names"] = list(example_input)
                    torch.onnx.export(model, args=(), kwargs=example_input, f=str(onnx_path), **export_kwargs)
            else:
                torch.onnx.export(model, example_input, str(onnx_path), **export_kwargs)
            print(f"[onnx] {name}: {onnx_path}", flush=True)
        ov_model = ov.convert_model(str(onnx_path))
    # OpenVINO 前端对部分 PyTorch 分支会保留动态维度。这里按导出样例
    # 固化所有输入，避免 Arc GPU 首次编译时出现动态 shape/显存抖动。
    shape_values: list[tuple[int, ...]] = []
    if isinstance(example_input, dict):
        shape_values = [tuple(int(v) for v in value.shape) for value in example_input.values()]
    elif isinstance(example_input, (tuple, list)):
        shape_values = [tuple(int(v) for v in value.shape) for value in example_input]
    elif hasattr(example_input, "shape"):
        shape_values = [tuple(int(v) for v in example_input.shape)]
    if shape_values and len(shape_values) == len(ov_model.inputs):
        ov_model.reshape({port: shape for port, shape in zip(ov_model.inputs, shape_values)})
    # Transformers Grounding DINO exports auxiliary decoder tensors as well.
    # Keep only the two public detection tensors: [B, queries, tokens] logits
    # and [B, queries, 4] normalized boxes.
    if name == "grounding_dino" and len(ov_model.outputs) != len(output_names):
        def dims(port: object) -> list[int]:
            try:
                return [int(value) for value in port.shape]
            except Exception:
                return []

        boxes = [port for port in ov_model.outputs if len(dims(port)) == 3 and dims(port)[-1] == 4]
        logits = [port for port in ov_model.outputs if len(dims(port)) == 3 and dims(port)[-1] not in {1, 2, 4}]
        if not boxes or not logits:
            raise RuntimeError(f"{name} 无法从 {len(ov_model.outputs)} 个输出中识别 boxes/logits")
        # Rebuild from graph nodes plus original Parameters. Passing Output
        # objects or inputs here selects an incompatible OpenVINO overload.
        ov_model = ov.Model(
            [logits[0].get_node(), boxes[0].get_node()],
            ov_model.get_parameters(),
            name,
        )
    if len(ov_model.outputs) != len(output_names):
        raise RuntimeError(f"{name} 输出数量异常：{len(ov_model.outputs)}，预期 {len(output_names)}")
    for port, output_name in zip(ov_model.outputs, output_names):
        port.get_tensor().set_names({output_name})
    # Grounding DINO 的分类 logits 会在文本编码器中产生较大的负值。
    # 压成 FP16 后这些值会溢出成 -inf，GroundingDinoProcessor 会把所有
    # 候选框过滤掉，最终表现为“请求成功但什么也没识别到”。检测模型保留
    # FP32；SAM2/RTMPose 仍使用 FP16 以降低显存占用。
    ov.save_model(ov_model, xml, compress_to_fp16=name != "grounding_dino")
    print(f"[done] {name}: {xml}")


def export_grounding_dino(force: bool = False) -> None:
    target = MODEL_DIR / "grounding-dino-tiny"
    required = ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt"]
    if not all((target / file).exists() for file in required):
        snapshot_download(repo_id=GROUNDING_REPO, local_dir=target, allow_patterns=required)
    processor = AutoProcessor.from_pretrained(target, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(target, local_files_only=True).eval()
    # transformers 4.49 uses an in-place boolean OR in this helper. PyTorch's
    # ONNX exporter cannot lower aten::__ior_, while this equivalent expression
    # is exportable and only affects this one-time fixed-shape conversion.
    from transformers.models.grounding_dino import modeling_grounding_dino

    def exportable_token_masks(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count = input_ids.shape
        special = torch.zeros((batch_size, token_count), device=input_ids.device, dtype=torch.bool)
        for token in modeling_grounding_dino.SPECIAL_TOKENS:
            special = torch.logical_or(special, input_ids == token)
        indexes = torch.nonzero(special)
        attention = torch.eye(token_count, device=input_ids.device, dtype=torch.bool).unsqueeze(0).repeat(batch_size, 1, 1)
        positions = torch.zeros((batch_size, token_count), device=input_ids.device)
        previous_column = 0
        for index in range(indexes.shape[0]):
            row, column = indexes[index]
            if (column == 0) or (column == token_count - 1):
                attention[row, column, column] = True
                positions[row, column] = 0
            else:
                attention[row, previous_column + 1 : column + 1, previous_column + 1 : column + 1] = True
                positions[row, previous_column + 1 : column + 1] = torch.arange(0, column - previous_column, device=input_ids.device)
            previous_column = column
        return attention, positions.to(torch.long)

    modeling_grounding_dino.generate_masks_with_special_tokens_and_transfer_map = exportable_token_masks
    inputs = processor(
        images=torch.zeros((3, 800, 1200), dtype=torch.uint8),
        text="person.",
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    # 固化文本长度，运行时也使用该 tokenizer padding；避免 GPU 插件动态 shape 编译抖动。
    fixed = {key: value for key, value in inputs.items()}
    save_ir(model, fixed, "grounding_dino", ("logits", "pred_boxes"), force=force)


class SAM2Encoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.sizes = [(256, 256), (128, 128), (64, 64)]

    def forward(self, image: torch.Tensor):
        backbone = self.model.forward_image(image)
        _, features, _, _ = self.model._prepare_backbone_features(backbone)
        if self.model.directly_add_no_mem_embed:
            features[-1] = features[-1] + self.model.no_mem_embed
        tensors = [feature.permute(1, 2, 0).reshape(1, -1, *size) for feature, size in zip(features[::-1], self.sizes[::-1])][::-1]
        return tensors[-1], tensors[0], tensors[1]


class SAM2Prompt(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.prompt = model.sam_prompt_encoder

    def forward(self, boxes: torch.Tensor):
        # 使用 SAM2 原生 box 分支。将框作为 [B, 4] 固定输入，避免 points
        # 分支中的布尔 index_put_，该算子无法稳定转换为 OpenVINO/ONNX。
        sparse, dense = self.prompt(points=None, boxes=boxes.reshape(1, 4), masks=None)
        return sparse, dense


class SAM2Decoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.decoder = model.sam_mask_decoder
        self.prompt = model.sam_prompt_encoder

    def forward(self, image_embed: torch.Tensor, high_res_0: torch.Tensor, high_res_1: torch.Tensor, sparse_embeddings: torch.Tensor, dense_embeddings: torch.Tensor):
        masks, _, _, _ = self.decoder(
            image_embeddings=image_embed,
            image_pe=self.prompt.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
        )
        return masks


def export_sam2_small() -> None:
    checkpoint = ensure_file(SAM2_REPO, SAM2_CHECKPOINT, MODEL_DIR / "sam2.1-hiera-small")
    from sam2.build_sam import build_sam2

    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint), device="cpu").eval()
    encoder = SAM2Encoder(model).eval()
    image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    save_ir(encoder, image, "sam2_encoder", ("image_embed", "high_res_0", "high_res_1"))

    prompt = SAM2Prompt(model).eval()
    boxes = torch.zeros((1, 4), dtype=torch.float32)
    save_ir(prompt, boxes, "sam2_prompt", ("sparse_embeddings", "dense_embeddings"))

    with torch.inference_mode():
        image_embed, high0, high1 = encoder(image)
        sparse, dense = prompt(boxes)
    decoder = SAM2Decoder(model).eval()
    save_ir(decoder, (image_embed, high0, high1, sparse, dense), "sam2_decoder", ("low_res_masks",))


def export_rtmpose() -> None:
    """下载官方 RTMPose-tiny ONNX 并转换固定 256x192 SimCC IR。"""
    onnx_path = IR_DIR / "rtmpose_tiny.onnx"
    xml = IR_DIR / "rtmpose_tiny.xml"
    if xml.exists() and xml.with_suffix(".bin").exists():
        print(f"[skip] rtmpose_tiny: {xml}")
        return
    IR_DIR.mkdir(parents=True, exist_ok=True)
    if not onnx_path.exists():
        archive = IR_DIR / "rtmpose_tiny_sdk.zip"
        print(f"[download] rtmpose_tiny: {RTMPOSE_SDK_URL}")
        urllib.request.urlretrieve(RTMPOSE_SDK_URL, archive)
        with zipfile.ZipFile(archive) as package:
            candidates = [name for name in package.namelist() if name.lower().endswith(".onnx")]
            if not candidates:
                raise RuntimeError("RTMPose 官方 SDK zip 中没有 ONNX 文件")
            # zip 可能同时携带动态和静态文件；优先选择 end2end/模型主文件。
            selected = next((name for name in candidates if "end2end" in name.lower()), candidates[0])
            with package.open(selected) as source, onnx_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        archive.unlink(missing_ok=True)
    model = ov.convert_model(str(onnx_path))
    # SDK 模型通常保留 batch 动态维度；本服务逐个 ROI 推理，固定为 1。
    model.reshape({port: [1, 3, 256, 192] for port in model.inputs})
    if len(model.outputs) != 2:
        raise RuntimeError(f"RTMPose ONNX outputs unexpected count: {len(model.outputs)}")
    model.outputs[0].get_tensor().set_names({"simcc_x"})
    model.outputs[1].get_tensor().set_names({"simcc_y"})
    ov.save_model(model, xml, compress_to_fp16=True)
    print(f"[done] rtmpose_tiny: {xml}")


def validate_ir() -> None:
    core = ov.Core()
    for xml in sorted(IR_DIR.glob("*.xml")):
        model = core.read_model(xml)
        print(f"[validate] {xml.name}: inputs={[port.get_any_name() for port in model.inputs]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local three-model OpenVINO IR")
    parser.add_argument("--clean", action="store_true", help="Delete old OpenVINO IR before export")
    parser.add_argument("--only", choices=("grounding_dino", "sam2_small", "rtmpose_tiny"), help="Export only one model")
    parser.add_argument("--force", action="store_true", help="Re-export the selected IR even when it already exists")
    args = parser.parse_args()
    if args.clean and IR_DIR.exists():
        shutil.rmtree(IR_DIR)
    exporters = {"grounding_dino": export_grounding_dino, "sam2_small": export_sam2_small, "rtmpose_tiny": export_rtmpose}
    selected = ((args.only, exporters[args.only]),) if args.only else tuple(exporters.items())
    for name, exporter in selected:
        try:
            print(f"[start] {name}", flush=True)
            if name == "grounding_dino":
                exporter(args.force)
            else:
                exporter()
            print(f"[ok] {name}", flush=True)
        except Exception as exc:
            print(f"[failed] {name}: {type(exc).__name__}: {exc}", flush=True)
            raise
    validate_ir()
    print("OpenVINO IR 导出完成。运行时可断网启动 main.py。")


if __name__ == "__main__":
    main()
