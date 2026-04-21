# chunk_md.py
import os
import re
import json
import argparse
import bisect
import time

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_single_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.md':
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .md 文件")

    print(f"正在读取文件: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            content = f.read()

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name, content

def extract_image_paths(text):
    """
    提取文本中所有 Markdown 图片路径，返回列表。
    匹配格式: ![alt](path) 或 ![alt](path "title")
    """
    pattern = r'!\[[^\]]*\]\(([^\s\)]+)(?:\s+["\'][^"\']*["\'])?\)'
    matches = re.findall(pattern, text)
    return matches  # 返回列表，可能为空

def get_line_number(pos, line_starts):
    idx = bisect.bisect_right(line_starts, pos) - 1
    return idx + 1

def split_text_by_tables(text):
    """
    将文本分割为普通段落和表格段落（包括 Markdown 表格和 HTML 表格）。
    返回列表，每个元素为 (segment_text, is_table)
    表格段落作为一个整体，不会被进一步切分。
    """
    lines = text.splitlines()
    segments = []
    i = 0
    n = len(lines)

    def is_md_table_start(idx):
        """判断是否为 Markdown 表格的开始行（包含表头和分隔行）"""
        if idx >= n - 1:
            return False
        line = lines[idx].strip()
        next_line = lines[idx + 1].strip()
        # 当前行必须包含 '|' 且不是纯分隔线
        if '|' not in line or re.fullmatch(r'[\s|:-]+', line):
            return False
        # 下一行必须是分隔线（如 |---|---|）
        if re.fullmatch(r'[\s|:-]+', next_line) and '|' in next_line:
            cells = [c.strip() for c in next_line.strip('|').split('|')]
            if any(re.fullmatch(r':?-{3,}:?', c) for c in cells if c):
                return True
        return False

    def is_html_table_start(idx):
        """判断是否为 HTML 表格的开始行（<table 标签）"""
        # 允许 <table 带属性，大小写不敏感
        return re.search(r'<\s*table', lines[idx], re.IGNORECASE) is not None

    def collect_md_table(start_idx):
        """收集完整的 Markdown 表格（从 start_idx 开始，直到连续非表格行）"""
        end_idx = start_idx
        while end_idx < n:
            line = lines[end_idx].strip()
            # 空行或完全不包含 '|' 且不是纯分隔线的行，视为表格结束
            if not line:
                break
            if '|' not in line and not re.fullmatch(r'[\s|:-]+', line):
                break
            end_idx += 1
        # 去掉末尾可能的多余空行
        while end_idx > start_idx and not lines[end_idx-1].strip():
            end_idx -= 1
        table_text = '\n'.join(lines[start_idx:end_idx])
        return end_idx, table_text

    def collect_html_table(start_idx):
        """收集完整的 HTML 表格（支持嵌套，使用栈计数）"""
        depth = 1
        end_idx = start_idx + 1
        # 找到对应的 </table>，注意大小写和属性
        while end_idx < n and depth > 0:
            line_lower = lines[end_idx].lower()
            # 开始标签 <table
            if re.search(r'<\s*table', line_lower):
                depth += 1
            # 结束标签 </table>
            if re.search(r'<\s*/\s*table\s*>', line_lower):
                depth -= 1
            end_idx += 1
        table_text = '\n'.join(lines[start_idx:end_idx])
        return end_idx, table_text

    while i < n:
        line_stripped = lines[i].strip()
        if is_md_table_start(i):
            end, table_text = collect_md_table(i)
            segments.append((table_text, True))
            i = end
        elif is_html_table_start(i):
            end, table_text = collect_html_table(i)
            segments.append((table_text, True))
            i = end
        else:
            start = i
            # 普通段落：累积直到遇到表格开始行
            while i < n and not is_md_table_start(i) and not is_html_table_start(i):
                i += 1
            para_text = '\n'.join(lines[start:i])
            if para_text.strip():
                segments.append((para_text, False))
    return segments

def parse_markdown_hierarchy(content, chunk_size, doc_name, source_file, file_id, file_version_id):
    lines_with_breaks = content.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for line in lines_with_breaks:
        line_starts.append(pos)
        pos += len(line)

    titles = []
    in_code_block = False
    for idx, line in enumerate(lines_with_breaks):
        stripped = line.rstrip('\n').lstrip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
        if not in_code_block and stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            if level >= 1:
                start_pos = line_starts[idx]
                end_pos = start_pos + len(line)
                title_text = stripped.strip()
                titles.append((start_pos, end_pos, level, title_text))

    # 如果没有标题，整个文档作为一块处理（仍支持表格保护）
    if not titles:
        all_chunks = []
        block_content = content
        # 对整个文档进行表格分割
        segments = split_text_by_tables(block_content)
        chunk_id = 0
        for seg_text, is_table in segments:
            if not seg_text.strip():
                continue
            # 表格段落：不切分；普通段落：若超长则切分
            if is_table:
                chunk_obj = {
                    "id": str(chunk_id),
                    "chunk_name": doc_name,
                    "content": seg_text,
                    "chapter": "",
                    "section": "",
                    "subsection": "",
                    "section_path": "0.0.0",
                    "source": get_line_number(0, line_starts),
                    "file": source_file,
                    "chunk_id": str(chunk_id),
                    "file_id": file_id,
                    "file_version_id": file_version_id,
                    "is_active": True,
                    "chunk_uid": f"{file_version_id}::{chunk_id}",
                    "table": True          # 标记为表格块
                }
                img_paths = extract_image_paths(seg_text)
                if img_paths:
                    chunk_obj["image_paths"] = img_paths
                all_chunks.append(chunk_obj)
                chunk_id += 1
            else:
                # 普通段落可能超长，需要切分
                if len(seg_text) <= chunk_size:
                    chunk_obj = {
                        "id": str(chunk_id),
                        "chunk_name": doc_name,
                        "content": seg_text,
                        "chapter": "",
                        "section": "",
                        "subsection": "",
                        "section_path": "0.0.0",
                        "source": get_line_number(0, line_starts),
                        "file": source_file,
                        "chunk_id": str(chunk_id),
                        "file_id": file_id,
                        "file_version_id": file_version_id,
                        "is_active": True,
                        "chunk_uid": f"{file_version_id}::{chunk_id}"
                    }
                    img_paths = extract_image_paths(seg_text)
                    if img_paths:
                        chunk_obj["image_paths"] = img_paths
                    all_chunks.append(chunk_obj)
                    chunk_id += 1
                else:
                    # 按行切分普通段落（尽量保持句子完整）
                    lines = seg_text.splitlines(keepends=True)
                    current_lines = []
                    current_len = 0
                    for line in lines:
                        line_len = len(line)
                        if current_len + line_len > chunk_size and current_lines:
                            chunk_text = ''.join(current_lines)
                            if chunk_text.strip():
                                chunk_obj = {
                                    "id": str(chunk_id),
                                    "chunk_name": doc_name,
                                    "content": chunk_text,
                                    "chapter": "",
                                    "section": "",
                                    "subsection": "",
                                    "section_path": "0.0.0",
                                    "source": get_line_number(0, line_starts),
                                    "file": source_file,
                                    "chunk_id": str(chunk_id),
                                    "file_id": file_id,
                                    "file_version_id": file_version_id,
                                    "is_active": True,
                                    "chunk_uid": f"{file_version_id}::{chunk_id}"
                                }
                                img_paths = extract_image_paths(chunk_text)
                                if img_paths:
                                    chunk_obj["image_paths"] = img_paths
                                all_chunks.append(chunk_obj)
                                chunk_id += 1
                            current_lines = []
                            current_len = 0
                        current_lines.append(line)
                        current_len += line_len
                    if current_lines:
                        chunk_text = ''.join(current_lines)
                        if chunk_text.strip():
                            chunk_obj = {
                                "id": str(chunk_id),
                                "chunk_name": doc_name,
                                "content": chunk_text,
                                "chapter": "",
                                "section": "",
                                "subsection": "",
                                "section_path": "0.0.0",
                                "source": get_line_number(0, line_starts),
                                "file": source_file,
                                "chunk_id": str(chunk_id),
                                "file_id": file_id,
                                "file_version_id": file_version_id,
                                "is_active": True,
                                "chunk_uid": f"{file_version_id}::{chunk_id}"
                            }
                            img_paths = extract_image_paths(chunk_text)
                            if img_paths:
                                chunk_obj["image_paths"] = img_paths
                            all_chunks.append(chunk_obj)
                            chunk_id += 1
        return all_chunks

    # 有标题的情况
    titles.append((len(content), len(content), 0, ""))
    all_chunks = []
    chunk_id = 0

    heading_counters = [0, 0, 0, 0, 0, 0, 0]
    cur_chapter_name = ""
    cur_section_name = ""
    cur_subsection_name = ""
    cur_chapter_num = 0
    cur_section_num = 0
    cur_subsection_num = 0

    for i in range(len(titles) - 1):
        title_start, title_end, level, title_text = titles[i]
        next_title_start = titles[i+1][0]

        # 更新标题计数
        if 2 <= level <= 6:
            heading_counters[level] += 1
            for j in range(level + 1, 7):
                heading_counters[j] = 0

        # 记录当前章节/小节名称
        if level == 1:
            cur_chapter_name = title_text.lstrip('#').strip()
        elif level == 2:
            cur_chapter_name = title_text.lstrip('#').strip()
            cur_chapter_num = heading_counters[2]
            cur_section_name = ""
            cur_section_num = 0
            cur_subsection_name = ""
            cur_subsection_num = 0
        elif level == 3:
            cur_section_name = title_text.lstrip('#').strip()
            cur_section_num = heading_counters[3]
            cur_subsection_name = ""
            cur_subsection_num = 0
        elif level == 4:
            cur_subsection_name = title_text.lstrip('#').strip()
            cur_subsection_num = heading_counters[4]
        elif level > 4:
            cur_subsection_name = title_text.lstrip('#').strip()

        content_start = title_end
        content_end = next_title_start
        full_block = content[content_start:content_end].lstrip('\n')
        if not full_block.strip():
            continue

        source_line = get_line_number(title_start, line_starts)
        # 对标题下的内容进行表格分割
        segments = split_text_by_tables(full_block)

        for seg_text, is_table in segments:
            if not seg_text.strip():
                continue

            if is_table:
                # 表格段落：完整保存，不切分
                chunk_obj = {
                    "id": str(chunk_id),
                    "chunk_name": doc_name,
                    "content": seg_text,
                    "chapter": cur_chapter_name,
                    "section": cur_section_name,
                    "subsection": cur_subsection_name,
                    "section_path": f"{cur_chapter_num}.{cur_section_num}.{cur_subsection_num}",
                    "source": source_line,
                    "file": source_file,
                    "chunk_id": str(chunk_id),
                    "file_id": file_id,
                    "file_version_id": file_version_id,
                    "is_active": True,
                    "chunk_uid": f"{file_version_id}::{chunk_id}",
                    "table": True          # 标记为表格块
                }
                img_paths = extract_image_paths(seg_text)
                if img_paths:
                    chunk_obj["image_paths"] = img_paths
                all_chunks.append(chunk_obj)
                chunk_id += 1
            else:
                # 普通段落：可能超过 chunk_size 需要切分
                if len(seg_text) <= chunk_size:
                    chunk_obj = {
                        "id": str(chunk_id),
                        "chunk_name": doc_name,
                        "content": seg_text,
                        "chapter": cur_chapter_name,
                        "section": cur_section_name,
                        "subsection": cur_subsection_name,
                        "section_path": f"{cur_chapter_num}.{cur_section_num}.{cur_subsection_num}",
                        "source": source_line,
                        "file": source_file,
                        "chunk_id": str(chunk_id),
                        "file_id": file_id,
                        "file_version_id": file_version_id,
                        "is_active": True,
                        "chunk_uid": f"{file_version_id}::{chunk_id}"
                    }
                    img_paths = extract_image_paths(seg_text)
                    if img_paths:
                        chunk_obj["image_paths"] = img_paths
                    all_chunks.append(chunk_obj)
                    chunk_id += 1
                else:
                    lines = seg_text.splitlines(keepends=True)
                    current_lines = []
                    current_len = 0
                    for line in lines:
                        line_len = len(line)
                        if current_len + line_len > chunk_size and current_lines:
                            chunk_text = ''.join(current_lines)
                            if chunk_text.strip():
                                chunk_obj = {
                                    "id": str(chunk_id),
                                    "chunk_name": doc_name,
                                    "content": chunk_text,
                                    "chapter": cur_chapter_name,
                                    "section": cur_section_name,
                                    "subsection": cur_subsection_name,
                                    "section_path": f"{cur_chapter_num}.{cur_section_num}.{cur_subsection_num}",
                                    "source": source_line,
                                    "file": source_file,
                                    "chunk_id": str(chunk_id),
                                    "file_id": file_id,
                                    "file_version_id": file_version_id,
                                    "is_active": True,
                                    "chunk_uid": f"{file_version_id}::{chunk_id}"
                                }
                                img_paths = extract_image_paths(chunk_text)
                                if img_paths:
                                    chunk_obj["image_paths"] = img_paths
                                all_chunks.append(chunk_obj)
                                chunk_id += 1
                            current_lines = []
                            current_len = 0
                        current_lines.append(line)
                        current_len += line_len
                    if current_lines:
                        chunk_text = ''.join(current_lines)
                        if chunk_text.strip():
                            chunk_obj = {
                                "id": str(chunk_id),
                                "chunk_name": doc_name,
                                "content": chunk_text,
                                "chapter": cur_chapter_name,
                                "section": cur_section_name,
                                "subsection": cur_subsection_name,
                                "section_path": f"{cur_chapter_num}.{cur_section_num}.{cur_subsection_num}",
                                "source": source_line,
                                "file": source_file,
                                "chunk_id": str(chunk_id),
                                "file_id": file_id,
                                "file_version_id": file_version_id,
                                "is_active": True,
                                "chunk_uid": f"{file_version_id}::{chunk_id}"
                            }
                            img_paths = extract_image_paths(chunk_text)
                            if img_paths:
                                chunk_obj["image_paths"] = img_paths
                            all_chunks.append(chunk_obj)
                            chunk_id += 1

    return all_chunks

def process_single_document_flow(input_file_path, chunk_size, file_id, file_version_id):
    doc_name, content = load_single_file(input_file_path)
    if not content:
        print("内容为空，跳过处理。")
        return []
    all_chunks = parse_markdown_hierarchy(
        content, chunk_size, doc_name, os.path.basename(input_file_path), file_id, file_version_id
    )
    return all_chunks

def append_to_json_array(file_path, new_objects):
    """将 new_objects 列表追加到 file_path 的 JSON 数组中，保持数组格式"""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []
    existing.extend(new_objects)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Markdown文档分块处理工具（支持表格整体保护）')
    parser.add_argument('--input', '-i', required=True, help='输入Markdown文件的完整路径 (.md)')
    parser.add_argument('--output', '-o', required=True, help='输出JSON文件路径（完整数组格式）')
    parser.add_argument('--chunk_size', '-s', type=int, default=800, help='分块大小（字符数）')
    parser.add_argument('--file_id', required=True, help='文件ID')
    parser.add_argument('--file_version_id', required=True, help='文件版本ID')
    parser.add_argument('--total_output', '-to', help='追加输出的JSON数组文件路径（标准JSON数组格式）', default=None)
    args = parser.parse_args()

    start_time = time.time()

    if os.path.isdir(args.input):
        print(f"❌ 错误: 输入路径 '{args.input}' 是一个文件夹，请提供具体的文件路径。")
        return

    chunks = process_single_document_flow(args.input, args.chunk_size, args.file_id, args.file_version_id)
    if chunks:
        save_json(chunks, args.output)
        print(f"✅ 文档分块完成，结果已保存到 {args.output}")
        print(f"📊 共生成 {len(chunks)} 个文本块")

        if args.total_output:
            os.makedirs(os.path.dirname(os.path.abspath(args.total_output)), exist_ok=True)
            append_to_json_array(args.total_output, chunks)
            print(f"✅ 已将 {len(chunks)} 个块追加到 {args.total_output} (JSON数组格式)")
    else:
        print("⚠️ 未生成任何分块结果")

    elapsed = time.time() - start_time
    print(f"\n=== 文档分块耗时: {elapsed:.2f} 秒 ===")

if __name__ == "__main__":
    main()