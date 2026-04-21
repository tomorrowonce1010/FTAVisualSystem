# run.py
from __future__ import annotations

"""
完整流水线：
1. 生成模式：PDF -> Markdown -> 清理标题层级 -> 文档分块 -> 实体提取 -> 关系提取
2. 导入模式：直接从已有产物目录读取 chunks / entities / relations，不执行生成步骤

修改说明（2026-04-21）：
- 分块结果除保存到版本目录外，还会追加到全局 output/chunk.json
- 实体提取和关系提取改为读写全局文件（output/entities.jsonl, output/entities_merged.json, output/relations.jsonl, output/relations.csv）
"""

import argparse
import os
import re
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Set

SCRIPT_DIR = Path(__file__).parent.absolute()


def run_command(cmd, description):
    """执行 shell 命令，安全处理 UTF-8 输出，并记录耗时。返回 subprocess.CompletedProcess。"""
    start = time.time()
    print(f"\n>>> {description}")
    print(f"命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = time.time() - start
    print(f"耗时: {elapsed:.2f} 秒")

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        print(f"错误: {description} 失败 (返回码 {result.returncode})")
        if stderr:
            print("stderr:", stderr)
        if stdout:
            print("stdout:", stdout)
        sys.exit(1)

    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    return result


def find_md_file(version_dir: Path, pdf_stem: str) -> Path:
    """在版本目录中查找生成的 .md 文件，并等待其写入完成。"""
    candidate = version_dir / f"{pdf_stem}.md"
    if not candidate.exists():
        candidate = version_dir / "ocr" / f"{pdf_stem}.md"
    if not candidate.exists():
        md_files = list(version_dir.rglob("*.md"))
        if md_files:
            candidate = md_files[0]
        else:
            raise FileNotFoundError(f"未在 {version_dir} 中找到任何 .md 文件")

    for _ in range(30):
        if candidate.exists() and candidate.stat().st_size > 100:
            return candidate
        time.sleep(1)

    raise RuntimeError(f"MD 文件生成失败或为空: {candidate}")


def get_latest_version_dir(output_root: Path, file_id: str) -> Path:
    """获取指定 file_id 下版本号最大的版本目录，例如 output_root/file_id/file_id_vN"""
    base_dir = output_root / file_id
    if not base_dir.exists():
        raise FileNotFoundError(f"未找到 file_id 目录: {base_dir}")

    pattern = re.compile(rf"^{re.escape(file_id)}_v(\d+)$")
    max_version = -1
    latest_dir = None
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                ver = int(match.group(1))
                if ver > max_version:
                    max_version = ver
                    latest_dir = item
    if latest_dir is None:
        raise FileNotFoundError(f"在 {base_dir} 下未找到任何版本目录 (格式: {file_id}_vN)")
    return latest_dir


def _discover_pdf_stem(import_only_dir: Path, explicit_stem: str | None) -> str:
    if explicit_stem:
        return explicit_stem

    chunk_files = sorted(import_only_dir.glob("*_chunks.json"))
    if len(chunk_files) == 1:
        return chunk_files[0].name[: -len("_chunks.json")]

    raise ValueError("导入模式下请提供 --pdf-stem，或保证目录中只有一个 *_chunks.json 文件")


def _resolve_artifacts(import_only_dir: Path, pdf_stem: str):
    return {
        "chunks_json": import_only_dir / f"{pdf_stem}_chunks.json",
        "entities_jsonl": import_only_dir / f"{pdf_stem}_entities.jsonl",
        "entities_merged_json": import_only_dir / f"{pdf_stem}_entities_merged.json",
        "relations_jsonl": import_only_dir / f"{pdf_stem}_relations.jsonl",
        "relations_csv": import_only_dir / f"{pdf_stem}_relations.csv",
    }


def _run_import_only_mode(args) -> None:
    import_only_dir = Path(args.import_only_dir).resolve()
    if not import_only_dir.exists():
        print(f"错误: 导入目录不存在 {import_only_dir}")
        sys.exit(1)

    explicit_stem = args.pdf_stem or (Path(args.pdf).stem if args.pdf else None)
    try:
        pdf_stem = _discover_pdf_stem(import_only_dir, explicit_stem)
    except ValueError as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    artifacts = _resolve_artifacts(import_only_dir, pdf_stem)
    if not artifacts["chunks_json"].exists():
        print(f"错误: 缺少 chunks 文件 {artifacts['chunks_json']}")
        sys.exit(1)
    if not args.skip_entity and not artifacts["entities_merged_json"].exists():
        print(f"错误: 缺少合并实体文件 {artifacts['entities_merged_json']}")
        sys.exit(1)
    if not args.skip_relation and not artifacts["relations_jsonl"].exists():
        print(f"错误: 缺少关系文件 {artifacts['relations_jsonl']}")
        sys.exit(1)

    print("\n=== 导入模式：直接复用已有产物，不执行生成步骤 ===")
    print(f"产物目录: {import_only_dir}")
    print(f"pdf_stem: {pdf_stem}")
    print(f"chunks: {artifacts['chunks_json']}")
    print(f"entities_merged: {artifacts['entities_merged_json']}")
    print(f"relations_jsonl: {artifacts['relations_jsonl']}")
    print("\n说明：本模式只验证并暴露已有产物路径，供上层服务自动同步到 generate-fta。")


# ========== 新增：全局文件操作 ==========
def append_chunks_to_global(chunks: List[Dict], global_chunk_path: Path) -> None:
    """
    将新生成的 chunks 追加到全局 chunk.json（JSON 数组）。
    自动去重：基于 chunk_uid 避免重复追加。
    """
    if not chunks:
        return

    # 读取现有全局 chunk 的 chunk_uid 集合
    existing_uids: Set[str] = set()
    if global_chunk_path.exists() and global_chunk_path.stat().st_size > 0:
        try:
            with open(global_chunk_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                for item in existing_data:
                    uid = item.get("chunk_uid")
                    if uid:
                        existing_uids.add(uid)
        except (json.JSONDecodeError, ValueError):
            # 文件损坏，重新初始化
            existing_uids = set()

    # 筛选新 chunk
    new_chunks = [c for c in chunks if c.get("chunk_uid") not in existing_uids]
    if not new_chunks:
        print("全局 chunk.json 已包含所有新分块，无需追加")
        return

    # 合并写入
    all_chunks = []
    if global_chunk_path.exists() and global_chunk_path.stat().st_size > 0:
        try:
            with open(global_chunk_path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            if not isinstance(all_chunks, list):
                all_chunks = []
        except json.JSONDecodeError:
            all_chunks = []
    all_chunks.extend(new_chunks)

    # 写回
    global_chunk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(global_chunk_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"已追加 {len(new_chunks)} 个新分块到全局文件: {global_chunk_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF 知识抽取完整流水线：PDF -> MD -> 清理MD -> 分块 -> 实体 -> 关系"
    )
    parser.add_argument("--pdf", "-p", help="输入的 PDF 文件路径；导入模式下仅用于推断 pdf_stem")
    parser.add_argument(
        "--pdf-stem",
        help="显式指定产物前缀（pdf_stem）。导入模式下用于定位 *_chunks.json；生成模式下可用于兼容上层服务参数契约。"
    )
    parser.add_argument("--output-dir", "-o", default="./output", help="输出根目录（默认 ./output）")
    parser.add_argument("--chunk-size", "-s", type=int, default=800, help="分块大小（字符数），默认 800")
    parser.add_argument("--skip-mineru", action="store_true", help="跳过 MinerU 转换步骤（假设已有 MD 文件）")
    parser.add_argument("--skip-clean", action="store_true", help="跳过 Markdown 标题层级清理步骤")
    parser.add_argument("--skip-entity", action="store_true", help="跳过实体提取（只执行到分块）")
    parser.add_argument("--skip-relation", action="store_true", help="跳过关系统取（只执行到实体合并）")
    parser.add_argument("--print-raw-text", action="store_true", help="打印 LLM 返回的原始文本（用于调试）")
    parser.add_argument("--import-only-dir", help="直接从此目录复用已有的 chunks/实体/关系产物，跳过所有生成步骤")
    args = parser.parse_args()

    if args.import_only_dir:
        _run_import_only_mode(args)
        return

    if not args.pdf:
        print("错误: 生成模式下必须提供 --pdf")
        sys.exit(1)

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"错误: PDF 文件不存在 {pdf_path}")
        sys.exit(1)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 全局产物路径
    global_chunk_path = output_root / "chunk.json"
    global_entities_jsonl = output_root / "entities.jsonl"
    global_entities_merged = output_root / "entities_merged.json"
    global_relations_jsonl = output_root / "relations.jsonl"
    global_relations_csv = output_root / "relations.csv"

    pdf_stem = pdf_path.stem

    # === 整个流水线开始时间 ===
    pipeline_start = time.time()

    # 确定版本目录和 MD 文件路径
    if not args.skip_mineru:
        print("\n=== 步骤1: PDF 转 Markdown (MinerU) ===")
        mineru_script = SCRIPT_DIR / "trans_file_to_md.py"
        if not mineru_script.exists():
            print(f"错误: 找不到 {mineru_script}，请确保脚本位于同一目录")
            sys.exit(1)

        cmd = [
            sys.executable,
            str(mineru_script),
            "-i",
            str(pdf_path),
            "-o",
            str(output_root),
            "-b",
            "pipeline",
            "-m",
            "ocr",
        ]
        result = run_command(cmd, "MinerU 转换")

        # 从输出中解析 VERSION_DIR
        version_dir = None
        for line in (result.stdout or "").splitlines():
            s = line if isinstance(line, str) else line.decode("utf-8", errors="replace")
            if s.startswith("VERSION_DIR="):
                version_dir = Path(s.split("=", 1)[1].strip())
                break
        if version_dir is None or not version_dir.exists():
            try:
                version_dir = get_latest_version_dir(output_root, pdf_stem)
                print(f"未从 MinerU 输出解析到 VERSION_DIR，使用最新版本目录: {version_dir}")
            except FileNotFoundError as e:
                print(f"错误: {e}")
                sys.exit(1)

        try:
            md_file = find_md_file(version_dir, pdf_stem)
            print(f"找到 Markdown 文件: {md_file} (大小: {md_file.stat().st_size} 字节)")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"错误: {exc}")
            sys.exit(1)
    else:
        # 跳过 MinerU，需要找到已有的版本目录
        if args.pdf_stem:
            file_id = args.pdf_stem
        else:
            file_id = pdf_stem
        try:
            version_dir = get_latest_version_dir(output_root, file_id)
            print(f"使用最新版本目录: {version_dir}")
        except FileNotFoundError as e:
            print(f"错误: {e}")
            sys.exit(1)

        try:
            md_file = find_md_file(version_dir, pdf_stem)
            print(f"使用现有 Markdown 文件: {md_file}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"错误: 找不到有效的 MD 文件 - {exc}")
            sys.exit(1)

    # 版本化知识库：file_version_id 必须是「整份文件版本」的稳定 ID
    file_version_id = version_dir.name

    # 后续所有产物都保存在版本目录中（分块备份 + MD/清理文件）
    result_dir = version_dir

    if not args.skip_clean:
        print("\n=== 步骤1.5: 清理 Markdown 标题层级 ===")
        clean_script = SCRIPT_DIR / "clean_md.py"
        if not clean_script.exists():
            print(f"错误: 找不到 {clean_script}")
            sys.exit(1)

        cleaned_md_file = result_dir / f"{pdf_stem}_cleaned.md"
        cmd_clean = [
            sys.executable,
            str(clean_script),
            "--input", str(md_file),
            "--output", str(cleaned_md_file)
        ]
        run_command(cmd_clean, "清理MD标题层级")
        md_file = cleaned_md_file
        print(f"清理后的 Markdown 文件: {md_file}")
    else:
        print("\n=== 跳过 Markdown 标题层级清理 ===")

    print("\n=== 步骤2: Markdown 分块 ===")
    chunk_script = SCRIPT_DIR / "chunk_md.py"
    if not chunk_script.exists():
        print(f"错误: 找不到 {chunk_script}")
        sys.exit(1)

    # 版本目录中的备份分块文件
    backup_chunks_json = result_dir / f"{pdf_stem}_chunks.json"
    cmd_chunk = [
        sys.executable,
        str(chunk_script),
        "--input", str(md_file),
        "--output", str(backup_chunks_json),
        "--chunk_size", str(args.chunk_size),
        "--file_id", pdf_stem,
        "--file_version_id", file_version_id,
    ]
    run_command(cmd_chunk, "文档分块")
    print(f"分块结果备份至: {backup_chunks_json}")

    # 读取刚刚生成的分块结果，追加到全局 chunk.json
    with open(backup_chunks_json, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    append_chunks_to_global(chunks_data, global_chunk_path)

    if args.skip_entity:
        print("已跳过实体提取，流程结束。")
        pipeline_elapsed = time.time() - pipeline_start
        print(f"\n=== 整个流水线耗时: {pipeline_elapsed:.2f} 秒 ===")
        return

    print("\n=== 步骤3: 实体提取 ===")
    entity_script = SCRIPT_DIR / "extract_entities.py"
    if not entity_script.exists():
        print(f"错误: 找不到 {entity_script}")
        sys.exit(1)

    cmd_entity = [
        sys.executable,
        str(entity_script),
        "--input", str(global_chunk_path),
        "--output-entities", str(global_entities_jsonl),
        "--output-merged", str(global_entities_merged),
    ]
    if args.print_raw_text:
        cmd_entity.append("--print-raw-text")
    run_command(cmd_entity, "实体提取")
    print(f"实体结果: {global_entities_jsonl}")
    print(f"合并实体: {global_entities_merged}")

    if args.skip_relation:
        print("已跳过关系统取，流程结束。")
        pipeline_elapsed = time.time() - pipeline_start
        print(f"\n=== 整个流水线耗时: {pipeline_elapsed:.2f} 秒 ===")
        return

    print("\n=== 步骤4: 关系提取 ===")
    relation_script = SCRIPT_DIR / "extract_relations.py"
    if not relation_script.exists():
        print(f"错误: 找不到 {relation_script}")
        sys.exit(1)

    cmd_relation = [
        sys.executable,
        str(relation_script),
        "--input-chunks", str(global_chunk_path),
        "--input-entities", str(global_entities_merged),
        "--output-relations", str(global_relations_jsonl),
        "--output-csv", str(global_relations_csv),
    ]
    if args.print_raw_text:
        cmd_relation.append("--print-raw-text")
    run_command(cmd_relation, "关系提取")
    print(f"关系 JSON: {global_relations_jsonl}")
    print(f"关系 CSV:  {global_relations_csv}")

    pipeline_elapsed = time.time() - pipeline_start
    print("\n=== 流水线执行完成 ===")
    print(f"整个流水线耗时: {pipeline_elapsed:.2f} 秒")
    print(f"结果存放目录: {result_dir}")
    print("生成文件（全局）:")
    print(f"  - 分块: {global_chunk_path}")
    print(f"  - 实体 (逐块): {global_entities_jsonl}")
    print(f"  - 实体 (合并): {global_entities_merged}")
    print(f"  - 关系 (逐块): {global_relations_jsonl}")
    print(f"  - 关系 (CSV):  {global_relations_csv}")
    print("备份文件（版本目录）:")
    print(f"  - 分块: {backup_chunks_json}")


if __name__ == "__main__":
    main()