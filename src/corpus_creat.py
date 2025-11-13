import json
import random
import os
import sqlite3
# ---------------- 配置 -----------------
val_id_path = "/home/hzm/MFC-LLM/data/PHM2012_data_val_ids.json"
output_json = "/home/hzm/MFC-LLM/data/PHM2012_data_corpus.json"
num_per_task = 500  # 每类任务生成数量

# 标签映射
label_map = {
    0: "Normal",
    1: "Fault"
}

# 任务名称映射
task_name_map = {
    0: "是否有故障",
    1: "故障类型识别",
    2: "剩余寿命预测",
    3: "运维建议",
    4: "综合分析(故障类型+剩余寿命+运维建议)"
}

# ---------------- 运维建议模板 -----------------
def generate_maintenance_advice(label_name, rul_value):
    templates = {
        "Normal": (
            f"Okay, the provided bearing condition is: **Fault-Free.**\n\n"
            f"**Estimated RUL:** {rul_value:.2f} units.\n\n"
            "**Analysis:**\n"
            "No detectable faults are present. The bearing is operating optimally.\n\n"
            "**Risks:**\n"
            "Since the bearing is fault-free, there are no immediate risks.\n\n"
            "**Potential Consequences:**\n"
            "No negative consequences from bearing defects are expected.\n\n"
            "**Recommended Actions:**\n"
            "* Regular vibration, temperature, and lubrication monitoring.\n"
            "* Maintain proper lubrication type, quantity, and schedule.\n"
            "* Ensure correct installation and alignment.\n"
            "* Prevent contamination (dust, moisture, debris).\n"
        ),
        "Fault": (
            f"Okay, the provided bearing condition is: **Inner Ring Fault detected.**\n\n"
            f"**Estimated RUL:** {rul_value:.2f} units.\n\n"
            "**Analysis:**\n"
            "Significant damage to the inner ring is detected, requiring immediate attention.\n\n"
            "**Risks:**\n"
            "High vibration, increased noise, elevated temperature, reduced load capacity, potential contamination.\n\n"
            "**Potential Consequences:**\n"
            "Complete bearing failure, shaft/housing damage, process interruption, safety hazards.\n\n"
            "**Recommended Actions:**\n"
            "* Immediate machine shutdown and inspection.\n"
            "* Replace the bearing promptly.\n"
            "* Perform root cause analysis for failure.\n"
            "* Check and maintain lubrication, alignment, and operating loads.\n"
        ),
        "Outer Race Fault": (
            f"Okay, the provided bearing condition is: **Outer Ring Fault detected.**\n\n"
            f"**Estimated RUL:** {rul_value:.2f} units.\n\n"
            "**Analysis:**\n"
            "Significant damage to the outer ring. Immediate maintenance is required.\n\n"
            "**Risks:**\n"
            "High vibration and noise, elevated temperature, reduced bearing capacity, potential catastrophic failure.\n\n"
            "**Potential Consequences:**\n"
            "Complete bearing failure, shaft/housing damage, downtime, safety hazards.\n\n"
            "**Recommended Actions:**\n"
            "* Immediate shutdown and inspection.\n"
            "* Replace the bearing without delay.\n"
            "* Review operating conditions, loads, alignment, and lubrication.\n"
        ),
        "Cage Fault": (
            f"Okay, the provided bearing condition is: **Cage Fault detected.**\n\n"
            f"**Estimated RUL:** {rul_value:.2f} units.\n\n"
            "**Analysis:**\n"
            "Severe damage to the bearing cage. Rolling elements may be misaligned.\n\n"
            "**Risks:**\n"
            "High vibration, noise, uneven load distribution, accelerated wear of inner/outer rings.\n\n"
            "**Potential Consequences:**\n"
            "Rapid bearing failure, machine downtime, potential damage to connected components.\n\n"
            "**Recommended Actions:**\n"
            "* Immediate inspection and maintenance.\n"
            "* Replace the bearing promptly.\n"
            "* Review lubrication, alignment, and operating conditions.\n"
        ),
        "Inner Race Fault and Outer Race Fault and Cage Fault and Ball Fault": (
            f"Okay, the provided bearing condition is: **Multiple Faults detected (Inner Race, Outer Race, Cage, Ball).**\n\n"
            f"**Estimated RUL:** {rul_value:.2f} units.\n\n"
            "**Analysis:**\n"
            "The bearing exhibits severe damage in multiple components, including inner race, outer race, cage, and rolling elements.\n\n"
            "**Risks:**\n"
            "Extremely high vibration, noise, uneven load distribution, rapid wear, potential catastrophic failure.\n\n"
            "**Potential Consequences:**\n"
            "Immediate complete bearing failure, severe damage to shaft/housing, unexpected downtime, and safety hazards.\n\n"
            "**Recommended Actions:**\n"
            "* Immediate machine shutdown and thorough inspection.\n"
            "* Replace the bearing immediately.\n"
            "* Conduct root cause analysis for multi-component failure.\n"
            "* Review lubrication, alignment, load, and operating conditions.\n"
        )
    }
    return templates.get(label_name, f"Unknown condition.\nEstimated RUL: {rul_value:.2f} units.")

def get_normal_ids_by_condition(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_id, condition_id FROM bearing_data WHERE label=0")
    normal_dict = {}
    for file_id, cond_id in cursor.fetchall():
        normal_dict.setdefault(cond_id, []).append(file_id)
    conn.close()
    if not normal_dict:
        raise ValueError("No normal samples found in SQLite database!")
    return normal_dict

# ---------------- 生成任务 -----------------
def generate_tasks(data, num_per_task=500, selected_tasks=[1,2]):
    corpus = []
    all_ids = list(range(len(data)))
    random.shuffle(all_ids)

    # 按 condition_id 保存正常样本
    normal_dict = {}
    for entry in data:
        fid = entry['file_id']
        cid = entry['condition_id']
        label = entry['label']
        if label == 0:
            normal_dict.setdefault(cid, []).append(fid)

    task_counter = 0
    task_samples = {tid: random.sample(all_ids, min(num_per_task, len(all_ids))) for tid in range(5)}

    def get_random_ref_id(condition_id):
        candidates = normal_dict.get(condition_id, [])
        return random.choice(candidates) if candidates else None

    # 单独生成每个任务，保留原逻辑
    if 0 in selected_tasks:
        print(f"正在生成任务 0 - {task_name_map[0]}")
        for idx in task_samples[0]:
            entry = data[idx]
            corpus.append({
                "id": task_counter,
                "task_id": 0,
                "instruction": "Based on the provided bearing state description #state_place_holder#, assess whether the bearing is experiencing any faults. Provide 'yes' or 'no'.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "ref_id": entry["ref_id"],
                "response": "no" if entry['label']==0 else "yes",
                "condition_id": entry['condition_id'],
                "rul": entry['rul']
            })
            task_counter += 1

    if 1 in selected_tasks:
        print(f"正在生成任务 1 - {task_name_map[1]}")
        for idx in task_samples[1]:
            entry = data[idx]
            corpus.append({
                "id": task_counter,
                "task_id": 1,
                "instruction": "Based on the provided bearing state description #state_place_holder#, identify the type of fault from [Normal,Fault].",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "ref_id": entry["ref_id"],
                "response": label_map[entry['label']],
                "condition_id": entry['condition_id'],
                "rul": entry['rul']
            })
            task_counter += 1

    if 2 in selected_tasks:
        print(f"正在生成任务 2 - {task_name_map[2]}")
        for idx in task_samples[2]:
            entry = data[idx]
            corpus.append({
                "id": task_counter,
                "task_id": 2,
                "instruction": "Based on the provided bearing state description #state_place_holder#, predict the remaining useful life (RUL) of the bearing.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "ref_id": entry["ref_id"],
                "response": f"The estimated remaining useful life (RUL) is {entry['rul']:.2f} units.",
                "condition_id": entry['condition_id'],
                "rul": entry['rul']
            })
            task_counter += 1

    if 3 in selected_tasks:
        print(f"正在生成任务 3 - {task_name_map[3]}")
        for idx in task_samples[3]:
            entry = data[idx]
            corpus.append({
                "id": task_counter,
                "task_id": 3,
                "instruction": "Based on the provided bearing state description #state_place_holder#, provide detailed maintenance advice for the bearing.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "ref_id": entry["ref_id"],
                "response": generate_maintenance_advice(label_map[entry['label']], entry['rul']),
                "condition_id": entry['condition_id'],
                "rul": entry['rul']
            })
            task_counter += 1

    if 4 in selected_tasks:
        print(f"正在生成任务 4 - {task_name_map[4]}")
        for idx in task_samples[4]:
            entry = data[idx]
            corpus.append({
                "id": task_counter,
                "task_id": 4,
                "instruction": "Based on the provided bearing state description #state_place_holder#, identify the type of fault from [Normal ,Fault], then predict the remaining useful life (RUL) of the bearing, and provide detailed maintenance advice for the bearing.",
                "label_id": entry['label'],
                "vib_id": entry['file_id'],
                "ref_id": entry["ref_id"],
                "response": generate_maintenance_advice(label_map[entry['label']], entry['rul']),
                "condition_id": entry['condition_id'],
                "rul": entry['rul']
            })
            task_counter += 1

    return corpus

# ---------------- 主函数 -----------------
if __name__ == "__main__":
    print("Loading validation set data...")
    with open(val_id_path, 'r') as f:
        data = json.load(f)
    print(f"Total samples in validation set: {len(data)}")

    print("Generating tasks...")
    corpus = generate_tasks(data, num_per_task=num_per_task)
    print(f"Total generated corpus size: {len(corpus)}")

    # 保存到 JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(corpus, f, indent=4)
    print(f"Corpus saved to {output_json}")
