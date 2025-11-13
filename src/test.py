import sys
import os
import torch
import numpy as np
import pandas as pd
import h5pickle as h5py
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer
import random
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fine_tuning import description_len, signal_token_id, get_bearllm, mod_xt_for_qwen

qwen_weights = '/home/hzm/MFC-LLM/LLM/qwen_weight'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
   0: "Normal",
   1: "Fault"
}

def parse_label(output_text: str) -> int:
    if "Normal" in output_text:
        return 0
    elif "Fault" in output_text:
        return 1
    else:
        return -1  # 未匹配

def run_classification(model, tokenizer, vib_data, ref_data, instruction):
    rv = np.array([vib_data, ref_data])
    np.save('./cache.npy', rv)

    place_holder_ids = torch.ones(description_len, dtype=torch.long) * signal_token_id
    text_part1, text_part2 = mod_xt_for_qwen(instruction)

    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids]).to(device)

    attention_mask = torch.ones_like(user_ids).to(device)

    with torch.no_grad():
        output = model.generate(
            user_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_new_tokens=64
        )

    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    return output_text

def run_rul_regression(model, vib_data, ref_data):
    rv = np.array([vib_data, ref_data])
    np.save('./cache.npy', rv)
    adapter = model.get_input_embeddings().adapter
    with torch.no_grad():
        vib_tensor = torch.tensor(rv, dtype=torch.float32).unsqueeze(0).to(device)
        rul_pred = adapter.predict_rul(vib_tensor)
    return rul_pred.item()

def evaluate_from_json(hdf5_path, test_json_path, weights_path, save_csv_path, sample_ratio=None, random_sample=True):
    with open(test_json_path, 'r') as f:
        data_list = json.load(f)

    if sample_ratio is not None and 0 < sample_ratio < 1:
        n_samples = int(len(data_list) * sample_ratio)
        if random_sample:
            data_list = random.sample(data_list, n_samples)
        else:
            data_list = data_list[:n_samples]

    with h5py.File(hdf5_path, 'r') as f:
        vib_dataset = f['vibration']

        tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = get_bearllm(train_mode=False)
        model = PeftModel.from_pretrained(base_model, weights_path)
        model.to(device)
        model.eval()

        results = []
        all_true_labels, all_pred_labels = [], []
        rul_errors = []

        for entry in tqdm(data_list):
            vib_id = entry['file_id']
            ref_id = entry['ref_id']
            condition_id = entry['condition_id']
            true_label = entry['label']
            true_rul = entry['rul']

            vib_data = vib_dataset[str(vib_id)][:]
            ref_data = vib_dataset[str(ref_id)][:]

            instruction_label = "Based on the provided bearing state description #state_place_holder#, identify the type of fault from [Normal Condition, Inner Race Fault, Outer Race Fault, Cage Fault, Inner Race Fault and Outer Race Fault and Cage Fault and Ball Fault]."
            output_label = run_classification(model, tokenizer, vib_data, ref_data, instruction_label)
            pred_label = parse_label(output_label)

            pred_rul = run_rul_regression(model, vib_data, ref_data)

            results.append({
                "vib_id": vib_id,
                "true_label": label_map[true_label],
                "pred_label": label_map.get(pred_label, "Unknown"),
                "true_rul": true_rul,
                "pred_rul": pred_rul,
                "output_label_text": output_label,
            })

            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)
            rul_errors.append(abs(pred_rul - true_rul))

        result_df = pd.DataFrame(results)
        result_df.to_csv(save_csv_path, index=False)
        print(f"已保存每个样本预测结果到 {save_csv_path}")

        acc = np.mean([t == p for t, p in zip(all_true_labels, all_pred_labels) if p != -1])
        mean_rul_error = np.mean(rul_errors) if rul_errors else None

        print("分类准确率: {:.2f}%".format(acc * 100))
        if mean_rul_error is not None:
            print("平均RUL误差: {:.4f}".format(mean_rul_error))
        else:
            print("没有成功预测出 RUL 数值")

if __name__ == "__main__":
    hdf5_path = "/home/hzm/MFC-LLM/data/PHM2012_data.hdf5"
    test_json_path = '/home/hzm/MFC-LLM/data/PHM2012_data_test_ids.json'
    weights_path = "/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/lora_PHM2012/checkpoint-2500"
    save_csv_path = "/home/hzm/MFC-LLM/data/PHM2012_test_results.csv"
    sample_ratio = 1
    evaluate_from_json(hdf5_path, test_json_path, weights_path, save_csv_path, sample_ratio)
