"""ModelScope SDK model downloader and local asset audit."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("vision-annotator")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR", str(PROJECT_DIR / "weights"))).resolve()
MODELSCOPE_CACHE_DIR = Path(os.getenv("MODELSCOPE_CACHE_DIR", str(WEIGHTS_DIR / ".modelscope_cache"))).resolve()
LOCK_FILE = WEIGHTS_DIR / "model_checksums.json"
DOWNLOAD_RETRIES = max(1, int(os.getenv("MODELSCOPE_DOWNLOAD_RETRIES", "3")))
# 健康接口每 5 秒轮询一次；重复计算数 GB 权重目录的 MD5 会让请求超时。
# 启动/下载时仍执行完整审计，轮询期间复用短时缓存。
AUDIT_CACHE_SECONDS = max(1.0, float(os.getenv("MODEL_AUDIT_CACHE_SECONDS", "30")))
_AUDIT_CACHE: dict[str, dict[str, Any]] | None = None
_AUDIT_CACHE_AT = 0.0


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    model_id: str
    relative_dir: str
    repository: str
    required: bool = True
    revision: str = "master"
    md5: str = ""
    # 资产审计只认真正的模型文件，避免 README 或配置文件让缺失模型被
    # 误报为 available。required_files 中的路径均相对于目标目录。
    required_files: tuple[str, ...] = ()

    @property
    def model_id_env(self) -> str:
        return f"MODELSCOPE_{self.name.upper()}_MODEL_ID"

    @property
    def configured_model_id(self) -> str:
        return os.getenv(self.model_id_env, self.model_id).strip()

    @property
    def target_dir(self) -> Path:
        return WEIGHTS_DIR / self.relative_dir


# All remote acquisition goes through ModelScope SDK snapshot_download first;
# Git LFS is used only as an explicit fallback when the SDK cannot complete.
ARTIFACTS: tuple[ModelArtifact, ...] = (
    ModelArtifact("grounding_dino", "AI-ModelScope/GroundingDINO", "grounding_dino", "https://modelscope.cn/models/AI-ModelScope/GroundingDINO", required_files=("groundingdino_swint_ogc.pth",)),
    ModelArtifact("sam2_small", "AI-ModelScope/sam2-hiera-base-plus", "sam2-hiera-base-plus", "https://modelscope.cn/models/AI-ModelScope/sam2-hiera-base-plus", required_files=("sam2_hiera_base_plus.pt",)),
    ModelArtifact("rtmpose_hand", "litert-community/RTMPose-Hand-LiteRT", "rtmpose-hand-litert", "https://modelscope.cn/models/litert-community/RTMPose-Hand-LiteRT", required_files=("rtmhand_fp16.tflite",)),
    ModelArtifact("rtmpose_tiny", "litert-community/RTMPose-s-LiteRT", "rtmpose-s-litert", "https://modelscope.cn/models/litert-community/RTMPose-s-LiteRT", required_files=("rtmpose_s_fp16.tflite",)),
    ModelArtifact("paddleocr", "PaddlePaddle/PaddleOCR-VL", "paddleocr-vl", "https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL", required_files=("model.safetensors", "PP-DocLayoutV2/inference.pdiparams")),
    ModelArtifact("moondream2", "AI-ModelScope/moondream2", "moondream2", "https://modelscope.cn/models/AI-ModelScope/moondream2", required_files=("model.safetensors",)),
)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """用于去重的内容指纹；与 MD5 审计分开，避免改变既有校验结果。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_directory(path: Path) -> str:
    digest = hashlib.md5()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        if ".git" in item.relative_to(path).parts or item.name == ".modelscope_complete":
            continue
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode())
        digest.update(md5_file(item).encode())
    return digest.hexdigest()


def _load_lock() -> dict[str, str]:
    try:
        value = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_lock(values: dict[str, str]) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate_audit_cache() -> None:
    """模型下载、去重或外部更新后调用，确保下一次健康检查重新审计。"""
    global _AUDIT_CACHE, _AUDIT_CACHE_AT
    _AUDIT_CACHE = None
    _AUDIT_CACHE_AT = 0.0


def _environment_md5(name: str, configured: str) -> str:
    return os.getenv(f"MODEL_MD5_{name.upper()}", configured).strip().lower()


def _has_model_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for item in path.rglob("*"):
        relative_parts = item.relative_to(path).parts
        if ".git" in relative_parts:
            continue
        if item.is_file() and item.name not in {".modelscope_complete", ".gitignore"} and not item.name.endswith((".part", ".tmp")):
            return True
    return False


def _missing_required_files(path: Path, artifact: ModelArtifact) -> list[str]:
    """返回快照中缺少的关键权重文件；配置/README 不计入完整性。"""
    return [relative for relative in artifact.required_files if not (path / relative).is_file()]


def _asset_complete(path: Path, artifact: ModelArtifact) -> bool:
    return _has_model_files(path) and not _missing_required_files(path, artifact)


def _snapshot_cache_dir(artifact: ModelArtifact) -> Path:
    """返回当前仓库 ID 对应的 SDK snapshot 目录。

    ModelScope 会在 ``cache_dir/models/<owner>--<name>/snapshots/<revision>``
    下保存快照。这里只使用当前配置的仓库 ID，避免误把旧仓库缓存当成
    正确模型。
    """
    cache_name = artifact.configured_model_id.replace("/", "--")
    return MODELSCOPE_CACHE_DIR / "models" / cache_name / "snapshots" / artifact.revision


def _same_file_content(left: Path, right: Path) -> bool:
    if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    try:
        if os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return sha256_file(left) == sha256_file(right)


def _hardlink_replace(source: Path, destination: Path) -> bool:
    """用硬链接替换目标副本，失败时保持原文件不变。"""
    try:
        if os.path.samefile(source, destination):
            return False
    except OSError:
        pass
    temporary = destination.with_name(destination.name + ".dedupe.tmp")
    try:
        if temporary.exists():
            temporary.unlink()
        os.link(source, temporary)
        os.replace(temporary, destination)
        return True
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("无法为重复文件创建硬链接 %s <- %s: %s", destination, source, exc)
        return False


def dedupe_artifact(artifact: ModelArtifact) -> dict[str, Any]:
    """合并当前目标目录与对应 SDK 快照中的完全相同文件。

    只做可逆的硬链接替换：缓存文件继续保留，目标路径也继续保留，
    但两者共享同一份磁盘数据。不同磁盘/权限不支持硬链接时仅报告，
    不会删除或覆盖源文件。
    """
    target = artifact.target_dir
    snapshot = _snapshot_cache_dir(artifact)
    result: dict[str, Any] = {
        "model": artifact.name,
        "target": str(target),
        "snapshot": str(snapshot),
        "linked": 0,
        "saved_bytes": 0,
        "skipped": 0,
        "errors": [],
    }
    if not target.is_dir() or not snapshot.is_dir():
        result["errors"].append("目标目录或当前仓库 snapshot 不存在")
        return result
    for destination in target.rglob("*"):
        if not destination.is_file() or destination.name in {".modelscope_complete", ".gitignore"}:
            continue
        try:
            relative = destination.relative_to(target)
            source = snapshot / relative
            if not source.is_file() or not _same_file_content(source, destination):
                continue
            if _hardlink_replace(source, destination):
                result["linked"] += 1
                result["saved_bytes"] += destination.stat().st_size
            else:
                result["skipped"] += 1
        except OSError as exc:
            result["errors"].append(f"{destination}: {exc}")
    return result


def dedupe_all_artifacts() -> list[dict[str, Any]]:
    """对六项当前资产执行显式去重，不触碰旧仓库缓存或 OpenVINO IR。"""
    results = [dedupe_artifact(artifact) for artifact in ARTIFACTS]
    for result in results:
        print(
            f"[dedupe] {result['model']}: linked={result['linked']} "
            f"saved={result['saved_bytes']} bytes skipped={result['skipped']}",
            flush=True,
        )
        for error in result["errors"]:
            print(f"[dedupe] {result['model']}: {error}", flush=True)
    return results


def _snapshot_download(artifact: ModelArtifact) -> Path:
    try:
        from modelscope import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 modelscope，请执行: python -m pip install modelscope") from exc
    MODELSCOPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "model_id": artifact.configured_model_id,
        "cache_dir": str(MODELSCOPE_CACHE_DIR),
        "max_workers": int(os.getenv("MODELSCOPE_MAX_WORKERS", "4")),
    }
    if artifact.revision:
        kwargs["revision"] = artifact.revision
    print(f"[modelscope] downloading {artifact.name}: {artifact.configured_model_id}", flush=True)
    return Path(snapshot_download(**kwargs))


def _git_repository_url(artifact: ModelArtifact) -> str:
    # ModelScope's Git clone command uses the repository root, not the
    # browser page path ``/models/<owner>/<model>``.
    base = os.getenv("MODELSCOPE_GIT_BASE_URL", "https://www.modelscope.cn").rstrip("/")
    return f"{base}/{artifact.configured_model_id}.git"


def _git_lfs_clone(artifact: ModelArtifact) -> Path:
    """Fallback for installations where ModelScope SDK cannot fetch a snapshot."""
    try:
        from git import Repo  # type: ignore
    except ImportError as exc:
        raise RuntimeError("SDK 下载失败且缺少 GitPython，请执行: python -m pip install gitpython") from exc

    target = artifact.target_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Git LFS 目标目录已有未完成内容，请清理后重试: {target}")
    repo_url = _git_repository_url(artifact)
    print(f"[git-lfs] cloning {artifact.name}: {repo_url}", flush=True)
    try:
        repo = Repo.clone_from(repo_url, str(target), depth=1)
        # GitPython delegates to the installed git-lfs executable. This is
        # intentionally not an HTTP fallback and preserves LFS resumability.
        repo.git.lfs("pull")
    except Exception as exc:
        raise RuntimeError(
            f"Git LFS 下载失败：{artifact.name}。请确认 git 与 git-lfs 已加入 PATH。"
        ) from exc
    (target / ".modelscope_complete").write_text("git lfs snapshot complete\n", encoding="ascii")
    return target


def download_artifact(artifact: ModelArtifact) -> Path:
    target = artifact.target_dir
    if _asset_complete(target, artifact):
        print(f"[skip] {artifact.name}: {target}", flush=True)
        return target
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"[modelscope] {artifact.name}: attempt {attempt}/{DOWNLOAD_RETRIES}", flush=True)
            snapshot_dir = _snapshot_download(artifact)
            if not snapshot_dir.is_dir():
                raise RuntimeError(f"ModelScope snapshot directory does not exist: {snapshot_dir}")
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot_dir, target, dirs_exist_ok=True)
            (target / ".modelscope_complete").write_text("snapshot complete\n", encoding="ascii")
            return target
        except Exception as exc:
            last_error = exc
            logger.warning("ModelScope download %s failed (%d/%d): %s", artifact.name, attempt, DOWNLOAD_RETRIES, exc)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    logger.warning("ModelScope SDK failed for %s; falling back to Git LFS.", artifact.name)
    try:
        return _git_lfs_clone(artifact)
    except Exception as fallback_error:
        raise RuntimeError(
            f"{artifact.name} 下载失败：ModelScope SDK 与 Git LFS 均不可用；"
            f"SDK={last_error}；Git LFS={fallback_error}。"
        ) from fallback_error


def audit_models(auto_download: bool = False) -> dict[str, dict[str, Any]]:
    global _AUDIT_CACHE, _AUDIT_CACHE_AT
    now = time.monotonic()
    if not auto_download and _AUDIT_CACHE is not None and now - _AUDIT_CACHE_AT < AUDIT_CACHE_SECONDS:
        # 返回浅拷贝，调用方修改单个状态不会污染缓存。
        return {name: dict(item) for name, item in _AUDIT_CACHE.items()}
    stored = _load_lock()
    changed = False
    report: dict[str, dict[str, Any]] = {}
    for artifact in ARTIFACTS:
        target = artifact.target_dir
        error = ""
        if not _asset_complete(target, artifact) and auto_download:
            try:
                target = download_artifact(artifact)
            except Exception as exc:
                error = str(exc)
                logger.warning("模型 %s 不可用: %s", artifact.name, exc)
        missing_required = _missing_required_files(target, artifact)
        if not _asset_complete(target, artifact):
            reason = "本地模型快照不存在" if not _has_model_files(target) else "缺少关键模型文件: " + ", ".join(missing_required)
            report[artifact.name] = {"available": False, "checksum": "missing", "missing_files": missing_required, "path": str(target), "source": artifact.repository, "repository": artifact.model_id, "model_id": artifact.configured_model_id, "cache_dir": str(MODELSCOPE_CACHE_DIR), "expected_md5": _environment_md5(artifact.name, artifact.md5), "error": error or reason}
            continue
        actual = md5_directory(target)
        expected = _environment_md5(artifact.name, artifact.md5)
        if expected:
            valid, checksum, error = actual == expected, ("verified" if actual == expected else "failed"), ("MD5 不匹配" if actual != expected else "")
        else:
            valid, checksum = True, "unverified"
            if stored.get(artifact.relative_dir) != actual:
                stored[artifact.relative_dir] = actual
                changed = True
        report[artifact.name] = {"available": valid, "checksum": checksum, "md5": actual, "missing_files": [], "expected_md5": expected, "path": str(target), "source": artifact.repository, "repository": artifact.model_id, "model_id": artifact.configured_model_id, "cache_dir": str(MODELSCOPE_CACHE_DIR), "error": error}
    if changed:
        _save_lock(stored)
    _AUDIT_CACHE = {name: dict(item) for name, item in report.items()}
    _AUDIT_CACHE_AT = time.monotonic()
    return report


def startup_audit() -> dict[str, dict[str, Any]]:
    return audit_models(os.getenv("AUTO_DOWNLOAD_MODELS", "0").lower() in {"1", "true", "yes"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="审计/下载六项本地视觉模型资产")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="将目标目录与当前 ModelScope snapshot 中相同文件合并为硬链接",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="只审计本地文件，不在缺失时下载",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.dedupe:
        dedupe_all_artifacts()
    print(json.dumps(audit_models(auto_download=not args.no_download), ensure_ascii=False, indent=2))
