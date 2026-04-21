# extract_relations.py
import json
import os
import re
import argparse
import csv
import concurrent.futures
import sys
import time
from typing import List, Dict, Set
from collections import defaultdict
from generate_prompt_relation import generate_relation_prompt_and_context_second
from llm_caller_relation import call_llm, reset_token_usage, get_token_usage

def normalize_entity_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    name = name.replace('\u3000', ' ')
    name = name.replace('\uff08', '(').replace('\uff09', ')')
    name = name.replace('\uff1a', ':').replace('\uff0c', ',').replace('\uff1b', ';')
    name = re.sub(r'[。？！]+$', '', name)
    name = name.replace('\uff0b', '+').replace('\uff06', '&')
    name = re.sub(r'[–—－‐]', '-', name)
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1]
    if name.startswith('[') and name.endswith(']'):
        name = name[1:-1]
    if name.startswith('{') and name.endswith('}'):
        name = name[1:-1]
    if name.startswith('<') and name.endswith('>'):
        name = name[1:-1]
    name = re.sub(r'\s+\(', '(', name)
    name = re.sub(r'\(\s+', '(', name)
    name = re.sub(r'\s+\)', ')', name)
    name = re.sub(r'\)\s+', ')', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.upper()
    name = name.strip()
    return name

def parse_relations_second(relation_text: str) -> List[Dict]:
    if not relation_text or not relation_text.strip():
        return []
    try:
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', relation_text.strip(), flags=re.IGNORECASE)
        data = json.loads(cleaned)
        if isinstance(data, list):
            relations = []
            for item in data:
                if isinstance(item, dict) and 'entity1' in item and 'entity2' in item and 'relation_type' in item:
                    relations.append({
                        "entity1": item["entity1"],
                        "entity2": item["entity2"],
                        "relation_type": item["relation_type"]
                    })
            if relations:
                return relations
    except json.JSONDecodeError:
        pass

    relations = []
    lines = relation_text.strip().splitlines()
    angle_pattern = re.compile(r'<([^,>]+)[,，]\s*([^,>]+)[,，]\s*([^,>]+)[,，]\s*([^,>]+)>')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = angle_pattern.match(line)
        if match:
            entity1 = match.group(1).strip()
            relation_type = match.group(2).strip()
            entity2 = match.group(4).strip()
            relations.append({
                "entity1": entity1,
                "entity2": entity2,
                "relation_type": relation_type
            })
            continue
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) == 3:
                relations.append({
                    "entity1": parts[0].strip(),
                    "entity2": parts[1].strip(),
                    "relation_type": parts[2].strip()
                })
                continue
        if ',' in line and line.count(',') == 2:
            parts = line.split(',')
            if len(parts) == 3:
                relations.append({
                    "entity1": parts[0].strip(),
                    "entity2": parts[1].strip(),
                    "relation_type": parts[2].strip()
                })
                continue
        pattern = r'([^->,\t]+?)\s*(?:[-–—>]+|\s+关系\s*[:：]?\s*)?\s*([^->,\t]+?)\s*[-–—>]+\s*([^->,\t]+)'
        match = re.search(pattern, line)
        if match:
            relations.append({
                "entity1": match.group(1).strip(),
                "entity2": match.group(3).strip(),
                "relation_type": match.group(2).strip()
            })
    return relations

def build_entity_props(entity: Dict) -> Dict:
    props = {
        "name": entity.get("name") or entity.get("entity_name", ""),
        "normalized_name": entity.get("normalized_name") or entity.get("name") or entity.get("entity_name", ""),
        "entity_type": entity.get("entity_type", ""),
        "description": entity.get("description", ""),
        "errorLevel": entity.get("errorLevel", "中"),
        "priority": entity.get("priority", 1),
        "probability": entity.get("probability", 1e-05),
        "showProbability": entity.get("showProbability", entity.get("probability", 1e-05)),
        "rule": entity.get("rule", ""),
        "investigateMethod": entity.get("investigateMethod", ""),
        "repairMethod": entity.get("repairMethod", ""),
    }
    if "documents" in entity:
        props["documents"] = entity["documents"]
    if "source_chunk_ids" in entity:
        props["source_chunk_ids"] = entity["source_chunk_ids"]
    if "support_count" in entity:
        props["support_count"] = entity["support_count"]
    for k in ("file_id", "file_version_id"):
        v = entity.get(k)
        if v not in (None, ""):
            props[k] = v
    if "is_active" in entity:
        props["is_active"] = bool(entity["is_active"])
    return props

def save_relations_to_csv_second(relations: List[Dict], output_csv_path: str) -> None:
    if not relations:
        print("警告：没有关系数据可保存，CSV文件将只包含表头")
    fieldnames = set()
    for rel in relations:
        fieldnames.update(rel.keys())
    # 将 chunk_uid 放在前面
    preferred_order = ["chunk_uid", "entity1", "entity2", "relation_type", "entity1_type", "entity2_type"]
    fieldnames = [f for f in preferred_order if f in fieldnames] + [f for f in fieldnames if f not in preferred_order]
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in relations:
            row = {k: rel.get(k, "") for k in fieldnames if k not in ["entity1_props", "entity2_props"]}
            writer.writerow(row)
    print(f"已保存 {len(relations)} 条关系到 {output_csv_path}")

def load_processed_chunk_uids(output_file: str) -> Set[str]:
    if not os.path.exists(output_file):
        return set()
    with open(output_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    return {str(item.get("chunk_uid", "")) for item in data if item.get("chunk_uid")}

def call_llm_with_retry(prompt: str, context: str, mode: str, max_retries: int = 3) -> str:
    """带重试机制的 LLM 调用"""
    for attempt in range(1, max_retries + 1):
        try:
            result = call_llm(prompt, context, mode=mode)
            if result and result.strip():
                return result
            else:
                if attempt < max_retries:
                    print(f"LLM 返回空结果，第 {attempt} 次重试...")
                else:
                    print(f"LLM 返回空结果，已达最大重试次数 {max_retries}")
        except Exception as e:
            if attempt < max_retries:
                print(f"LLM 调用异常: {e}，第 {attempt} 次重试...")
            else:
                print(f"LLM 调用异常: {e}，已达最大重试次数 {max_retries}")
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))  # 1, 2, 4, ...
    return ""

def extract_relations_incremental(chunks: List[Dict], entities_results: List[Dict], output_file: str,
                                  print_raw_text: bool = False, max_workers: int = 10,
                                  max_retries: int = 3) -> List[Dict]:
    processed_chunk_uids = load_processed_chunk_uids(output_file)
    results = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = [json.loads(line.strip()) for line in f]

    chunk_entities_map = {res["chunk_uid"]: res["entities"] for res in entities_results}

    pending_chunks = []
    for chunk in chunks:
        chunk_uid = chunk.get("chunk_uid")
        if not chunk_uid:
            print(f"警告：chunk 缺少 chunk_uid 字段，跳过该 chunk: {chunk}")
            continue
        if chunk_uid in processed_chunk_uids:
            continue
        pending_chunks.append((chunk_uid, chunk))

    if not pending_chunks:
        print("没有需要处理的新 chunk")
        return results

    print(f"需要处理 {len(pending_chunks)} 个 chunk，使用多线程并发调用 LLM，并发数: {max_workers}，重试次数: {max_retries}")

    def process_one(chunk_uid: str, chunk: Dict):
        chunk_name = chunk.get("chunk_name") or "未知文档"
        content = chunk.get("content", "")
        
        file_id = chunk.get("file_id", "")
        file_version_id = chunk.get("file_version_id", "")
        is_active = chunk.get("is_active", True)
        file_name = chunk.get("file", "")
        
        title_parts = []
        if chunk.get("section_path"):
            title_parts.append(f"章节路径: {chunk['section_path']}")
        if chunk.get("chapter"):
            title_parts.append(f"章: {chunk['chapter']}")
        if chunk.get("section"):
            title_parts.append(f"节: {chunk['section']}")
        if chunk.get("subsection"):
            title_parts.append(f"小节: {chunk['subsection']}")
        
        if title_parts:
            title_str = "\n".join(title_parts)
            enriched_content = f"{title_str}\n\n{content}"
        else:
            enriched_content = content
        
        entities = chunk_entities_map.get(chunk_uid, [])
        
        original_entity_map = {}
        normalized_entity_map = {}
        for ent in entities:
            name = ent.get("name") or ent.get("entity_name")
            if name:
                original_entity_map[name] = ent
                norm_name = normalize_entity_name(name)
                if norm_name not in normalized_entity_map:
                    normalized_entity_map[norm_name] = ent
        
        print(f"处理文档关系：{chunk_name}，chunk_uid: {chunk_uid}，实体数: {len(entities)}")
        valid_relations = []
        try:
            if entities:
                relation_prompt, relation_context = generate_relation_prompt_and_context_second(
                    chunk_name, enriched_content, entities
                )
                relation_text = call_llm_with_retry(relation_prompt, relation_context, mode="relation", max_retries=max_retries)

                if not relation_text:
                    print(f"警告：chunk {chunk_uid} LLM 调用最终失败，跳过该 chunk")
                    return None

                if print_raw_text:
                    print(f"\n=== LLM 原始返回 (chunk_uid: {chunk_uid}) ===\n{relation_text}\n=================================\n")

                relations = parse_relations_second(relation_text)
                for rel in relations:
                    norm_e1 = normalize_entity_name(rel["entity1"])
                    norm_e2 = normalize_entity_name(rel["entity2"])
                    
                    if norm_e1 in normalized_entity_map and norm_e2 in normalized_entity_map:
                        relation_type = rel["relation_type"]
                        ent1 = normalized_entity_map[norm_e1]
                        ent2 = normalized_entity_map[norm_e2]
                        
                        if relation_type in ("检测", "修复"):
                            if relation_type == "检测":
                                ent2.setdefault("detection_methods", []).append(ent1.get("name") or ent1.get("entity_name"))
                            elif relation_type == "修复":
                                ent2.setdefault("repair_methods", []).append(ent1.get("name") or ent1.get("entity_name"))
                        else:
                            props1 = build_entity_props(ent1)
                            props2 = build_entity_props(ent2)
                            
                            for props in (props1, props2):
                                props["file_id"] = file_id
                                props["file_version_id"] = file_version_id
                                props["is_active"] = is_active
                                source_ids = props.get("source_chunk_ids", [])
                                if source_ids:
                                    props["documents"] = [{"chunk_uid": cid} for cid in source_ids]
                                else:
                                    props["documents"] = []
                                raw_name = props.get("name", "")
                                props["normalized_name"] = normalize_entity_name(raw_name)
                            
                            rel["entity1_type"] = ent1.get("entity_type", "")
                            rel["entity2_type"] = ent2.get("entity_type", "")
                            rel["entity1_props"] = props1
                            rel["entity2_props"] = props2
                            
                            # 使用 chunk_uid 作为标识
                            rel["chunk_uid"] = chunk_uid
                            rel["file_id"] = file_id
                            rel["file_version_id"] = file_version_id
                            rel["file_name"] = file_name
                            rel["is_active"] = is_active
                            
                            valid_relations.append(rel)
                    else:
                        if print_raw_text:
                            print(f"关系实体未匹配: {rel['entity1']} -> {norm_e1}, {rel['entity2']} -> {norm_e2}")

            result = {
                "chunk_uid": chunk_uid,
                "file_id": file_id,
                "file_version_id": file_version_id,
                "file_name": file_name,
                "is_active": is_active,
                "relations": valid_relations,
            }
            print(f"chunk {chunk_uid} 提取有效关系数：{len(valid_relations)}")
            return result
        except Exception as e:
            print(f"处理 chunk_uid {chunk_uid} 关系时出错: {e}")
            return None

    new_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_one, cuid, chunk): cuid for cuid, chunk in pending_chunks}
        with open(output_file, "a", encoding="utf-8") as f:
            for future in concurrent.futures.as_completed(future_to_chunk):
                cuid = future_to_chunk[future]
                try:
                    result = future.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        new_results.append(result)
                except Exception as e:
                    print(f"处理 chunk {cuid} 时发生异常: {e}")

    results.extend(new_results)
    return results

def load_chunks(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line.strip()) for line in f if line.strip()]
    # 确保每个 chunk 都有 chunk_uid，若没有则尝试使用 id 或 chunk_id 生成警告
    for chunk in data:
        if "chunk_uid" not in chunk:
            # 兼容旧数据：尝试使用 id 或 chunk_id，但推荐使用 chunk_uid
            if "id" in chunk:
                chunk["chunk_uid"] = str(chunk["id"])
                print(f"警告：chunk 缺少 chunk_uid，使用 id 字段替代: {chunk['id']}")
            elif "chunk_id" in chunk:
                chunk["chunk_uid"] = str(chunk["chunk_id"])
                print(f"警告：chunk 缺少 chunk_uid，使用 chunk_id 字段替代: {chunk['chunk_id']}")
            else:
                print(f"错误：chunk 缺少 chunk_uid、id 和 chunk_id，无法处理: {chunk}")
                continue
    return data

def load_aggregated_entities(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line.strip()) for line in f if line.strip()]
    chunk_to_entities = defaultdict(list)
    for ent in data:
        entity_name = ent.get("name") or ent.get("entity_name")
        if not entity_name:
            continue
        ent["entity_name"] = entity_name
        if "name" not in ent:
            ent["name"] = entity_name
        chunk_ids = ent.get("source_chunk_ids", ent.get("chunk_ids", []))
        if not chunk_ids:
            continue
        for cid in chunk_ids:
            chunk_to_entities[str(cid)].append(ent)
    # 返回结构中的键改为 chunk_uid
    return [{"chunk_uid": cuid, "entities": ents} for cuid, ents in chunk_to_entities.items()]

def main():
    parser = argparse.ArgumentParser(description='实体关系提取工具 (基于 chunk_uid)')
    parser.add_argument('--input-chunks', '-ic', required=True, help='输入chunks JSON文件路径')
    parser.add_argument('--input-entities', '-ie', required=True, help='输入实体JSON文件路径（必须提供）')
    parser.add_argument('--output-relations', '-or', required=True, help='输出关系JSON文件路径')
    parser.add_argument('--output-csv', '-oc', required=True, help='输出CSV文件路径')
    parser.add_argument('--skip-relation-extraction', action='store_true', help='跳过关系统取，只进行CSV转换')
    parser.add_argument('--print-raw-text', '-p', action='store_true', help='打印LLM返回的原始关系文本（用于调试）')
    parser.add_argument('--max-workers', '-mw', type=int, default=10, help='并发线程数，默认为10')
    parser.add_argument('--max-retries', '-mr', type=int, default=3, help='LLM调用失败时的最大重试次数，默认为3')

    args = parser.parse_args()

    start_time = time.time()
    reset_token_usage()

    print(f"正在加载chunks数据: {args.input_chunks}")
    chunks = load_chunks(args.input_chunks)
    print(f"加载完成，共 {len(chunks)} 个chunks")

    if not args.input_entities or not os.path.exists(args.input_entities):
        print(f"错误: 必须提供有效的实体文件路径 (--input-entities)，当前: {args.input_entities}")
        sys.exit(1)

    print(f"使用外部实体文件: {args.input_entities}")
    entities_results = load_aggregated_entities(args.input_entities)
    print(f"从聚合实体文件生成 {len(entities_results)} 个chunk的实体数据")

    if not args.skip_relation_extraction:
        print(f"开始关系提取，结果将保存到: {args.output_relations}")
        relations_results = extract_relations_incremental(
            chunks, entities_results, args.output_relations,
            print_raw_text=args.print_raw_text,
            max_workers=args.max_workers,
            max_retries=args.max_retries
        )
        print(f"关系提取完成，共处理 {len(relations_results)} 个chunks")
    else:
        print("跳过关系统取，直接加载已有结果")
        if os.path.exists(args.output_relations):
            with open(args.output_relations, "r", encoding="utf-8") as f:
                relations_results = [json.loads(line.strip()) for line in f]
            print(f"加载已有关系结果: {len(relations_results)} 个chunks")
        else:
            print(f"错误: 关系结果文件不存在: {args.output_relations}")
            return

    print(f"开始转换为CSV，结果将保存到: {args.output_csv}")
    all_relations = []
    for result in relations_results:
        for rel in result.get("relations", []):
            all_relations.append(rel)
    save_relations_to_csv_second(all_relations, args.output_csv)
    print(f"CSV转换完成，共 {len(all_relations)} 个关系")

    elapsed = time.time() - start_time
    usage = get_token_usage()
    print(f"\n=== 关系提取耗时: {elapsed:.2f} 秒 ===")
    print(f"Token 用量: prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']}, total={usage['total_tokens']}")
    print(f"\n=== 统计信息 ===")
    print(f"总关系数量: {len(all_relations)}")

if __name__ == "__main__":
    main()