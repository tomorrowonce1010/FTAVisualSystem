# extract_entities.py
import json
import os
import re
import argparse
import time
from typing import List, Dict, Set, Any
from generate_prompt_relation import generate_entity_prompt_and_context
from llm_caller_relation import call_llm, reset_token_usage, get_token_usage
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from difflib import SequenceMatcher

ALLOWED_ENTITY_TYPES = {"故障原因与现象", "逻辑与"}

write_lock = threading.Lock()

def standardize_entity_name(name: str) -> str:
    if not name:
        return name
    s = name.strip()
    s = s.replace('\u3000', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('：', ':')
    if s.endswith('。'):
        s = s[:-1]
    s = s.replace('＋', '+').replace('＆', '&')
    s = re.sub(r'[–—－‑]', '-', s)
    pairs = [
        ('"', '"'), ("'", "'"),
        ('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'),
        ('（', '）'),
    ]
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if s.startswith(left) and s.endswith(right):
                s = s[1:-1]
                changed = True
                break
    s = s.upper()
    return s

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_entities(text: str, chunk: Dict) -> List[Dict]:
    try:
        data = json.loads(text)
        entities_data = data.get("entities", [])
    except json.JSONDecodeError:
        print("警告: LLM输出不是有效JSON，尝试按行解析")
        entities_data = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.search(r'实体名称[：:]\s*(.+?)\s*[，,]\s*实体类别[：:]\s*(.+)', line)
            if match:
                entities_data.append({
                    "name": match.group(1).strip(),
                    "entity_type": match.group(2).strip(),
                    "description": "",
                    "rule": "",
                    "investigateMethod": "",
                    "repairMethod": ""
                })

    doc_name = chunk.get("document_name") or chunk.get("chunk_name", "未知文档")
    source = chunk.get("source", "未知行")
    chunk_uid = str(chunk.get("chunk_uid") or chunk.get("chunk_id") or chunk.get("id") or "0")

    valid_entities = []
    seen_names = set()
    content = chunk.get("content", "")

    for ent in entities_data:
        raw_name = ent.get("name", "").strip()
        if not raw_name:
            continue
        name = standardize_entity_name(raw_name)
        if not name or name in seen_names:
            continue
        entity_type = ent.get("entity_type", "")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue
        if len(name) > 20:
            continue

        fid = str(chunk.get("file_id") or "").strip()
        fvid = str(chunk.get("file_version_id") or "").strip()
        entity_obj = {
            "name": name,
            "entity_type": entity_type,
            "description": ent.get("description", ""),
            "errorLevel": "中",
            "priority": 1,
            "probability": None,
            "showProbability": None,
            "rule": ent.get("rule", ""),
            "investigateMethod": ent.get("investigateMethod", ""),
            "repairMethod": ent.get("repairMethod", ""),
            "documents": [{
                "document_name": doc_name,
                "source": source
            }],
            "source_chunk_ids": [chunk_uid],
            "support_count": 1,
            "file_id": fid,
            "file_version_id": fvid,
            "is_active": bool(chunk.get("is_active", True)),
        }
        valid_entities.append(entity_obj)
        seen_names.add(name)

    return valid_entities

def is_entity_in_text(entity_name: str, text: str) -> bool:
    e = entity_name.strip().lower()
    t = text.lower()
    pattern = re.escape(e).replace(r'\ ', r'\s*')
    return re.search(pattern, t) is not None

def load_processed_chunk_uids(output_file: str) -> Set[str]:
    if not os.path.exists(output_file):
        return set()
    with open(output_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    return {item.get("chunk_uid", "") for item in data if item.get("chunk_uid")}

def process_single_chunk(chunk: Dict, print_raw_text: bool, write_lock, output_file: str, processed_chunk_uids: Set[str]):
    chunk_uid = str(chunk.get("chunk_uid") or chunk.get("chunk_id") or chunk.get("id") or "0")
    if chunk_uid in processed_chunk_uids:
        return None

    chunk_name = chunk.get("chunk_name") or "未知文档"
    content = chunk.get("content", "")

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

    print(f"处理文档：{chunk_name}，chunk_uid: {chunk_uid}")

    max_retries = 3
    entities = []
    for attempt in range(1, max_retries + 1):
        try:
            entity_prompt, entity_context = generate_entity_prompt_and_context(chunk_name, enriched_content)
            entity_text = call_llm(entity_context, entity_prompt)
            if print_raw_text:
                print(f"LLM output for chunk {chunk_uid} (attempt {attempt}):\n{entity_text}")

            entities = parse_entities(entity_text, chunk)
            if entities:
                break
            else:
                if attempt < max_retries and ('{' in entity_text or '[' in entity_text or '实体名称' in entity_text):
                    print(f"警告: chunk {chunk_uid} 第{attempt}次提取实体为空，但输出包含疑似JSON结构，将重试")
                else:
                    break
        except Exception as e:
            print(f"chunk {chunk_uid} 第{attempt}次调用出错: {e}")
            if attempt == max_retries:
                entities = []
            continue

    if not entities:
        print(f"警告: chunk {chunk_uid} 最终未能提取到有效实体（重试{max_retries}次）")

    result = {
        "chunk_uid": chunk_uid,
        "chunk_name": chunk_name,
        "content": content,
        "entities": entities
    }

    with write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"提取有效实体数：{len(entities)} (chunk {chunk_uid})")
    return result

def extract_entities_incremental(chunks: List[Dict], output_file: str, print_raw_text: bool = False, max_workers: int = 5) -> List[Dict]:
    processed_chunk_uids = load_processed_chunk_uids(output_file)
    results = []

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = [json.loads(line.strip()) for line in f]

    chunks_to_process = []
    for chunk in chunks:
        uid = str(chunk.get("chunk_uid") or chunk.get("chunk_id") or chunk.get("id") or "0")
        if uid not in processed_chunk_uids:
            chunks_to_process.append(chunk)

    if not chunks_to_process:
        print("所有chunk均已处理，无需提取")
        return results

    print(f"待处理chunk数：{len(chunks_to_process)}，并发数：{max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_single_chunk, chunk, print_raw_text, write_lock, output_file, processed_chunk_uids): chunk
            for chunk in chunks_to_process
        }

        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                uid = str(chunk.get("chunk_uid") or chunk.get("chunk_id") or chunk.get("id") or "0")
                print(f"处理chunk {uid}时发生未捕获异常: {e}")

    return results

# ================== 修改开始：实体合并相关函数（基于名称+描述） ==================

def compute_entity_similarity(e1: Dict, e2: Dict) -> float:
    """计算两个实体的相似度，基于名称和描述的拼接文本"""
    text1 = f"{e1['name']} {e1['description']}".strip()
    text2 = f"{e2['name']} {e2['description']}".strip()
    if not text1 or not text2:
        # 若描述为空，回退到仅名称
        text1 = e1['name']
        text2 = e2['name']
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def cluster_entities_by_similarity(entities: List[Dict], threshold: float = 0.8) -> List[List[Dict]]:
    """
    基于实体名称+描述的相似度进行聚类
    输入: entities 列表，每个元素需包含 name 和 description
    输出: 聚类后的实体组列表，每个组是实体字典的列表
    """
    used = set()
    clusters = []
    for i, e1 in enumerate(entities):
        if i in used:
            continue
        group = [e1]
        used.add(i)
        for j, e2 in enumerate(entities[i+1:], start=i+1):
            if j in used:
                continue
            if compute_entity_similarity(e1, e2) >= threshold:
                group.append(e2)
                used.add(j)
        clusters.append(group)
    return clusters

def llm_judge_equivalent_names(entity_infos: List[Dict]) -> tuple:
    """
    entity_infos: 每个元素为 {"name": str, "description": str}
    返回: (equivalent, standard_name)
    """
    if len(entity_infos) <= 1:
        return True, entity_infos[0]["name"] if entity_infos else ""

    items = []
    for idx, info in enumerate(entity_infos, 1):
        name = info["name"]
        desc = info["description"] if info["description"] else "无描述"
        items.append(f"实体{idx}: 名称={name}, 描述={desc}")

    prompt = f"""请判断以下技术实体是否指向同一个故障实体（即同义不同表述）。如果是，请给出一个最能代表该实体的标准名称（可以从列表中选择或适当合并）；如果不是，请输出"NOT_EQUIVALENT"。

实体列表：
{chr(10).join(items)}

要求：
- 结合名称和描述进行综合判断，只有当所有实体在技术含义上完全一致时才视为等价。
- 输出格式为JSON：{{"equivalent": true/false, "standard_name": "标准名称"}}（若equivalent为false，standard_name可为空字符串）。
- 只输出JSON，不要有其他解释。
"""
    context = "你是一个技术文档实体对齐专家，擅长结合名称和描述判断实体是否相同。"
    try:
        response = call_llm(context, prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data.get("equivalent", False), data.get("standard_name", "")
        else:
            return False, ""
    except Exception as e:
        print(f"LLM判断名称+描述等价失败: {e}")
        return False, ""

def generate_merge_prompt(instances: List[Dict]) -> str:
    instance_texts = []
    for idx, inst in enumerate(instances, 1):
        instance_texts.append(f"""实例 {idx}:
- 描述: {inst['description']}
- 排查规则: {inst['rule']}
- 调查方法: {inst['investigateMethod']}
- 修复方法: {inst['repairMethod']}
""")
    instances_block = "\n".join(instance_texts)

    prompt = f"""你是一个专业的技术文档整理专家。以下有多个描述同一个技术实体的文本片段（实体名称相同），它们可能来自不同文档或不同段落。请将这些片段中的信息合并成一份简洁、连贯、不重复的完整描述。

要求：
1. 对于“描述”字段：合并所有关键信息，去除重复，保留最完整、最准确的表述。
2. 对于“排查规则”、“调查方法”、“修复方法”字段：合并所有不重复的步骤或要点，可以按逻辑顺序重新组织，但不要遗漏重要信息。
3. 如果某个字段在所有实例中均为空，则输出空字符串。
4. 输出格式为严格的 JSON 对象，包含以下四个字段：
   {{
     "description": "合并后的描述",
     "rule": "合并后的排查规则",
     "investigateMethod": "合并后的调查方法",
     "repairMethod": "合并后的修复方法"
   }}
5. 只输出 JSON，不要有其他解释文字。

待合并的实体实例：
{instances_block}
"""
    return prompt

def merge_entities_with_llm(instances: List[Dict]) -> Dict:
    if not instances:
        return {"description": "", "rule": "", "investigateMethod": "", "repairMethod": ""}
    if len(instances) == 1:
        return {
            "description": instances[0]["description"],
            "rule": instances[0]["rule"],
            "investigateMethod": instances[0]["investigateMethod"],
            "repairMethod": instances[0]["repairMethod"],
        }

    merge_prompt = generate_merge_prompt(instances)
    context = "你是一个专业的技术文档整理助手，擅长合并多个来源的相同实体信息。"

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            response_text = call_llm(context, merge_prompt)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                merged = json.loads(json_match.group(0))
                for field in ["description", "rule", "investigateMethod", "repairMethod"]:
                    if field not in merged:
                        merged[field] = ""
                return merged
            else:
                raise ValueError("LLM 返回内容不包含有效 JSON")
        except Exception as e:
            print(f"LLM 合并实体失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                merged = {}
                for field in ["description", "rule", "investigateMethod", "repairMethod"]:
                    values = [inst[field] for inst in instances if inst[field]]
                    unique_values = []
                    for v in values:
                        if v not in unique_values:
                            unique_values.append(v)
                    merged[field] = "; ".join(unique_values)
                return merged
    return {"description": "", "rule": "", "investigateMethod": "", "repairMethod": ""}

def merge_entities(entities_results: List[Dict], output_file: str, max_workers: int = 5):
    name_to_instances = {}
    for result in entities_results:
        for entity in result["entities"]:
            name = entity["name"]
            if name not in name_to_instances:
                name_to_instances[name] = []
            instance = {
                "description": entity["description"],
                "rule": entity["rule"],
                "investigateMethod": entity["investigateMethod"],
                "repairMethod": entity["repairMethod"],
                "entity_type": entity["entity_type"],
                "errorLevel": entity["errorLevel"],
                "priority": entity["priority"],
                "probability": entity["probability"],
                "showProbability": entity["showProbability"],
                "documents": entity["documents"].copy(),
                "source_chunk_ids": set(entity["source_chunk_ids"]),
                "support_count": 1,
                "file_id": str(entity.get("file_id") or "").strip(),
                "file_version_id": str(entity.get("file_version_id") or "").strip(),
                "is_active": bool(entity.get("is_active", True)),
            }
            name_to_instances[name].append(instance)

    if not name_to_instances:
        save_json([], output_file)
        return

    print(f"合并前原始实体数量（不同标准化名称）: {len(name_to_instances)}")

    # 构建实体列表（用于聚类）
    entity_list = []
    for name, instances in name_to_instances.items():
        # 取第一个实例的描述作为代表（聚类阶段只做粗筛）
        first_desc = instances[0]["description"] if instances else ""
        entity_list.append({
            "name": name,
            "description": first_desc,
            "instances": instances  # 保存所有实例，用于后续合并
        })

    print(f"开始基于名称+描述的相似度聚类，共 {len(entity_list)} 个实体...")
    similarity_clusters = cluster_entities_by_similarity(entity_list, threshold=0.5)
    print(f"相似度聚类完成，生成 {len(similarity_clusters)} 个候选组")

    # 对每个候选组进行 LLM 等价判断（并发）
    final_groups = []
    name_to_standard = {}

    def process_equivalence_cluster(cluster_entities):
        # cluster_entities 是 List[Dict]，每个包含 name, description, instances
        if len(cluster_entities) == 1:
            return cluster_entities, True, cluster_entities[0]["name"]
        else:
            infos = [{"name": ent["name"], "description": ent["description"]} for ent in cluster_entities]
            equivalent, standard_name = llm_judge_equivalent_names(infos)
            return cluster_entities, equivalent, standard_name

    total_clusters = len(similarity_clusters)
    print(f"开始对 {total_clusters} 个候选组进行并发 LLM 等价判断（并发数={max_workers}）...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {
            executor.submit(process_equivalence_cluster, cluster): cluster
            for cluster in similarity_clusters
        }
        for idx, future in enumerate(as_completed(future_to_cluster), 1):
            if idx % 10 == 0 or idx == total_clusters:
                print(f"  等价判断进度: {idx}/{total_clusters}")
            cluster_entities, equivalent, standard_name = future.result()
            if equivalent and standard_name:
                final_groups.append(cluster_entities)      # 整个组合并为一个最终实体
                for ent in cluster_entities:
                    name_to_standard[ent["name"]] = standard_name
            else:
                # 不等价，每个实体独立成组
                for ent in cluster_entities:
                    final_groups.append([ent])
                    name_to_standard[ent["name"]] = ent["name"]

    print(f"开始合并文本字段，共 {len(final_groups)} 个实体组（并发数={max_workers}）...")

    def process_merge_group(group):
        all_instances = []
        for ent in group:
            all_instances.extend(ent["instances"])
        if not all_instances:
            return None
        standard_name = name_to_standard[group[0]["name"]]
        merged_texts = merge_entities_with_llm(all_instances)
        return {
            "standard_name": standard_name,
            "all_instances": all_instances,
            "merged_texts": merged_texts
        }

    merge_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_group = {
            executor.submit(process_merge_group, group): group
            for group in final_groups
        }
        for idx, future in enumerate(as_completed(future_to_group), 1):
            if idx % 10 == 0 or idx == len(final_groups):
                print(f"  文本合并进度: {idx}/{len(final_groups)}")
            result = future.result()
            if result:
                merge_results.append(result)

    merged_entities = []
    for res in merge_results:
        all_instances = res["all_instances"]
        first = all_instances[0]
        entity_type = first["entity_type"]
        errorLevel = first["errorLevel"]
        priority = first["priority"]
        probability = first["probability"]
        showProbability = first["showProbability"]

        # 合并 documents（去重，保持原有结构）
        merged_docs = []
        for inst in all_instances:
            for doc in inst["documents"]:
                if not any(d.get("document_name") == doc.get("document_name") and d.get("source") == doc.get("source") for d in merged_docs):
                    merged_docs.append(doc)

        # 合并 source_chunk_ids（集合转列表）
        merged_chunk_uids = set()
        for inst in all_instances:
            merged_chunk_uids.update(inst["source_chunk_ids"])

        # 合并 file_id 和 file_version_id（去重，保留非空值）
        merged_file_ids = set()
        merged_file_version_ids = set()
        for inst in all_instances:
            fid = inst.get("file_id", "")
            if fid:
                merged_file_ids.add(fid)
            fvid = inst.get("file_version_id", "")
            if fvid:
                merged_file_version_ids.add(fvid)

        merged_entity = {
            "name": res["standard_name"],
            "entity_type": entity_type,
            "description": res["merged_texts"]["description"],
            "errorLevel": errorLevel,
            "priority": priority,
            "probability": probability,
            "showProbability": showProbability,
            "rule": res["merged_texts"]["rule"],
            "investigateMethod": res["merged_texts"]["investigateMethod"],
            "repairMethod": res["merged_texts"]["repairMethod"],
            "documents": merged_docs,
            "source_chunk_ids": list(merged_chunk_uids),
            "support_count": len(merged_chunk_uids),
            "file_ids": list(merged_file_ids),
            "file_version_ids": list(merged_file_version_ids),
            "is_active": bool(first.get("is_active", True)),
        }
        merged_entities.append(merged_entity)

    save_json(merged_entities, output_file)

# ================== 修改结束 ==================

def main():
    parser = argparse.ArgumentParser(description='实体识别和合并工具（支持多线程并发，支持名称+描述等价合并）')
    parser.add_argument('--input', '-i', required=True, help='输入chunks JSON文件路径')
    parser.add_argument('--output-entities', '-oe', required=True, help='输出实体JSON文件路径（每行一个chunk结果）')
    parser.add_argument('--output-merged', '-om', required=True, help='输出合并实体JSON文件路径')
    parser.add_argument('--skip-entity-extraction', action='store_true', help='跳过实体提取，只进行合并')
    parser.add_argument('--print-raw-text', '-p', action='store_true', help='打印LLM返回的原始实体文本（用于调试）')
    parser.add_argument('--max-workers', '-w', type=int, default=10, help='并发线程数（实体提取和合并共用，默认10）')

    args = parser.parse_args()

    start_time = time.time()
    reset_token_usage()

    print(f"正在加载chunks数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载完成，共 {len(chunks)} 个chunks")

    if not args.skip_entity_extraction:
        print(f"开始实体提取（多线程，并发数={args.max_workers}），结果将保存到: {args.output_entities}")
        entities_results = extract_entities_incremental(chunks, args.output_entities,
                                                        args.print_raw_text, args.max_workers)
        print(f"实体提取完成，共处理 {len(entities_results)} 个chunks")
    else:
        print("跳过实体提取，直接加载已有结果")
        if os.path.exists(args.output_entities):
            with open(args.output_entities, "r", encoding="utf-8") as f:
                entities_results = [json.loads(line.strip()) for line in f]
            print(f"加载已有实体结果: {len(entities_results)} 个chunks")
        else:
            print(f"错误: 实体结果文件不存在: {args.output_entities}")
            return

    print(f"开始实体合并（含名称+描述等价判断和LLM智能合并，并发数={args.max_workers}），结果将保存到: {args.output_merged}")
    merge_entities(entities_results, args.output_merged, max_workers=args.max_workers)
    print("实体合并完成")

    with open(args.output_merged, "r", encoding="utf-8") as f:
        merged_entities = json.load(f)

    elapsed = time.time() - start_time
    usage = get_token_usage()
    print(f"\n=== 实体提取与合并耗时: {elapsed:.2f} 秒 ===")
    print(f"Token 用量: prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']}, total={usage['total_tokens']}")
    print(f"\n=== 统计信息 ===")
    print(f"合并后实体数量: {len(merged_entities)}")

if __name__ == "__main__":
    main()