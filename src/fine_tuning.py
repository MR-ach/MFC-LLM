import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from dotenv import dotenv_values
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import numpy as np
from F2LNet import FeatureEncoder
from datasets import CorpusDataset
from F2LNet import Classifier, LifeRegressor  
import torch.nn.functional as F

env = dotenv_values()
qwen_weights = '/home/hzm/MFC-LLM/LLM/qwen_weight'
fcn_weights = '/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/F2LNet_PHM2012' 
l3_weights = '/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/l3.npy'
align_weights = '/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/align.pth'
adapter_weights = '/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/vibration_adapter.pth'
description_len = 5
llm_hidden_size = 1536
signal_token_id = 151925
m = [4**4, 4**3, 4**2, 4**1, 1]

class HyperParameters:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r = 4
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.per_device_train_batch_size = 20
        self.gradient_accumulation_steps = 4
        self.logging_steps = 20
        self.num_train_epochs = 50
        self.save_steps = 2125
        self.learning_rate = 1e-4
        self.lr_scheduler_type = 'cosine'

description_text = [
    "Normal",
    "Fault"
]

description_tokens = [
    [220, 220, 220, 58780, 62890],
    [57024, 220, 36356, 21525, 59149],
    [68623, 349, 36356, 21525, 59149],
    [1514, 19289, 36356, 21525, 59149],
    [57024, 220, 12836, 220, 59149],
    [68623, 349, 12836, 220, 59149],
    [1514, 19289, 12836, 220, 59149],
    [57024, 220, 55197, 21525, 59149],
    [68623, 349, 55197, 21525, 59149],
    [1514, 19289, 55197, 21525, 59149]
]
                   #修改过
sys_prompt = (
    "As an expert in bearing fault diagnosis with extensive knowledge in mechanical engineering and failure "
    "analysis, you can assess the state of bearings. Based on the description of the bearing state, answer my questions."
)


def initialize_l3_weight():
    llm = AutoModelForCausalLM.from_pretrained(qwen_weights)
    embedding = llm.get_input_embeddings()
    tokens = torch.tensor(description_tokens)
    embeds = embedding(tokens).to(torch.float32).detach().cpu().numpy()
    np.save(l3_weights, embeds)
    return embeds

def load_l3_weight():
    if not os.path.exists(l3_weights):
        return initialize_l3_weight()
    return np.load(l3_weights)

class AlignmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 128) 
        self.relu = nn.ReLU()
        self.linear_cls = nn.Linear(128, 5)      
        self.linear_rul = nn.Linear(128, 1)       
        self.linear3 = nn.Linear(6, description_len * llm_hidden_size)  

    def forward(self, x):
        x = x.view(-1, 1024) 
        x = self.linear1(x)
        x = self.relu(x)
        cls_feat = self.linear_cls(x)
        rul_feat = self.linear_rul(x)
        combined = torch.cat([cls_feat, rul_feat], dim=1)  
        x = self.linear3(combined)
        x = x.view(x.size(0), description_len, llm_hidden_size)
        x = x.to(torch.bfloat16)
        return x

    def load_default(self):
        classifier_weights = torch.load(f'{fcn_weights}/classifier.pth', map_location='cpu')
        self.linear1.weight.data = classifier_weights['linear1.weight']
        self.linear1.bias.data = classifier_weights['linear1.bias']
        self.linear_cls.weight.data = classifier_weights['linear2.weight']
        self.linear_cls.bias.data = classifier_weights['linear2.bias']
        regressor_weights = torch.load(f'{fcn_weights}/regressor.pth', map_location='cpu')
        self.linear_rul.weight.data = regressor_weights['linear2.weight']
        self.linear_rul.bias.data = regressor_weights['linear2.bias']
        l3_weight = load_l3_weight()
        l3_weight = torch.from_numpy(l3_weight)
        l3_weight = l3_weight.reshape(l3_weight.size(0), -1)
        # if self.linear3.weight.shape == (l3_weight.numel(), 3):
        #     self.linear3.weight.data = l3_weight.reshape(-1, 3).T

    def save_weights(self):
        torch.save(self.state_dict(), align_weights)

    def load_weights(self, map_location='cpu'):
        self.load_state_dict(torch.load(align_weights, map_location=map_location))

@torch.no_grad()
def decode_sample_id(signal_ids_tensor):
    signal_ids_tensor = signal_ids_tensor.view(-1, 5)   
    signal_ids_tensor = signal_ids_tensor - signal_token_id
    m_t = torch.tensor(m, device=signal_ids_tensor.device).unsqueeze(0) 
    sample_ids = (signal_ids_tensor * m_t).sum(dim=1)  
    return sample_ids

class IdConverter:
    def __init__(self, train_mode=True):
        self.hp = HyperParameters()
        self.test_file = './cache.npy'
        self.dataset = CorpusDataset() if train_mode else None
    @torch.no_grad()
    def get_signal(self, signal_ids_tensor, train_mode=True):
        s = signal_ids_tensor
        if s.dim() == 1:
            s = s.unsqueeze(0)
        sample_ids = decode_sample_id(s)  # [N]
        res = []
        if train_mode:
            for sid in sample_ids:
                sid_int = int(sid.item()) if isinstance(sid, torch.Tensor) else int(sid)
                item = self.dataset.__getitem__(sid_int)
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    vib = item[2]
                else:
                    raise RuntimeError('CorpusDataset返回格式不符合预期')
                res.append(np.array(vib))
        else:
            arr = np.load(self.test_file)
            os.remove(self.test_file)
            res.append(arr)
        data = np.stack(res, axis=0)  
        return torch.tensor(data, dtype=torch.float32, device=self.hp.device)

class AlignmentAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.classifier = Classifier()
        self.regressor = LifeRegressor()
        self.alignment_layer = AlignmentLayer()

    def forward(self, x):
        feat = self.feature_encoder(x)           
        return self.alignment_layer(feat)  

    def predict_rul(self, x):
        feat = self.feature_encoder(x)
        rul = self.regressor(feat)
        return rul.view(-1)

    def save_weights(self):
        torch.save(self.state_dict(), adapter_weights)

    def load_default(self):
        self.feature_encoder.load_weights(fcn_weights)
        self.classifier.load_weights(fcn_weights)
        self.regressor.load_weights(fcn_weights)
        self.alignment_layer.load_default()

    def load_weights(self, map_location='cpu'):
        if not os.path.exists(adapter_weights):
            self.load_default()
            self.save_weights()
        else:
            self.load_state_dict(torch.load(adapter_weights, map_location=map_location))

class ModifiedEmbedding(nn.Module):
    def __init__(self, embedding, train_mode=True):
        super().__init__()
        self.embedding = embedding
        self.adapter = AlignmentAdapter()
        self.adapter.load_weights()
        self.adapter.to(embedding.weight.device)
        self.signal_converter = IdConverter(train_mode=train_mode)
        self._device = embedding.weight.device
        self._H = llm_hidden_size
        self._desc_len = description_len

    def forward(self, x):
        B, T = x.size()
        if (x >= signal_token_id).sum() == 0:
            return self.embedding(x)
        safe_ids = x.clone()
        safe_ids[safe_ids >= signal_token_id] = self.embedding.num_embeddings - 1  
        base = self.embedding(safe_ids)  
        base = base.to(torch.bfloat16)
        for b in range(B):
            mask = (x[b] >= signal_token_id)  
            num_sig = int(mask.sum().item())
            if num_sig == 0:
                continue
            if num_sig != self._desc_len:
                raise ValueError(f"Sample {b} has {num_sig} signal tokens, expected {self._desc_len}")
            signal_ids_flat = x[b, mask]  
            sig_ids_batch = signal_ids_flat.unsqueeze(0)
            signal_feats = self.signal_converter.get_signal(sig_ids_batch, self.training)  
            aligned = self.adapter(signal_feats)  
            aligned = aligned.to(torch.bfloat16)
            base[b, mask, :] = aligned[0]
        return base  
    @property
    def weight(self):
        return self.embedding.weight

def get_bearllm(train_mode=True):
    hp = HyperParameters()
    config = AutoConfig.from_pretrained(f'{qwen_weights}/config.json')
    model = AutoModelForCausalLM.from_pretrained(
        qwen_weights,
        device_map=hp.device,
        torch_dtype="auto",
        trust_remote_code=True,
        config=config
    )
    embedding = model.get_input_embeddings()
    mod_embedding = ModifiedEmbedding(embedding, train_mode=train_mode)
    model.set_input_embeddings(mod_embedding)
    return model

def mod_xt_for_qwen(xt):
    text_part1 = '<|im_start|>system\n' + sys_prompt + '\n<|im_end|><|im_start|>user\n' + xt.split('#state_place_holder#')[0]
    text_part2 = xt.split('#state_place_holder#')[1] + '<|im_end|>\n<|im_start|>assistant\n'
    return text_part1, text_part2

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)

    task_ids = torch.tensor([x.get('task_id', 1) for x in batch], dtype=torch.long)
    ruls = torch.tensor([x.get('rul', float('nan')) if x.get('rul', None) is not None else float('nan') for x in batch], dtype=torch.float32)
    rul_mask = (task_ids == 2)
    if rul_mask.any():
        labels_clone = labels.clone()
        labels_clone[rul_mask] = -100
        labels = labels_clone

    return {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'labels': labels.long(),
        'task_id': task_ids,
        'rul': ruls
    }

def encode_sample_id(x):
    result = []
    remainder = x
    for i in range(len(m)):
        digit = remainder // m[i]
        result.append(digit)
        remainder %= m[i]
    return torch.tensor(result, dtype=torch.int)

class FineTuningDataset(Dataset):
    def __init__(self):
        self.dataset = CorpusDataset()
        self.hp = HyperParameters()
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        if isinstance(item, (list, tuple)):
            if len(item) == 7:
                sample_id, task_id, label_id, vib, instruction, response, rul = item
            elif len(item) == 6:
                sample_id, label_id, vib, instruction, response, rul = item
                task_id = 2 if rul is not None else 1
            else:
                raise RuntimeError('CorpusDataset 返回长度不支持: ' + str(len(item)))
        else:
            raise RuntimeError('CorpusDataset 未返回可解析的序列')
        signal_ids = signal_token_id + encode_sample_id(sample_id)
        user_part1, user_part2 = mod_xt_for_qwen(instruction)
        user_part1_ids = self.tokenizer(user_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_part2_ids = self.tokenizer(user_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_ids = torch.cat([user_part1_ids, signal_ids, user_part2_ids])
        gt_ids = self.tokenizer(response, return_tensors='pt', add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([user_ids, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.cat([torch.ones_like(user_ids) * -100, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        if task_id is None:
            task_id = 2 if rul is not None else 1
        rul_value = float(rul) if (rul is not None and not (isinstance(rul, float) and np.isnan(rul))) else None
        return {
            'input_ids': input_ids.long().detach(),
            'attention_mask': attention_mask.long().detach(),
            'labels': labels.long().detach(),
            'task_id': int(task_id),
            'rul': torch.tensor(rul_value, dtype=torch.float32) if rul_value is not None else None
        }

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task_ids = inputs.pop('task_id')
        ruls = inputs.pop('rul')
        device = next(model.parameters()).device
        labels = inputs.get('labels', None)
        outputs = model(**{k: v for k, v in inputs.items() if k in ('input_ids', 'attention_mask', 'labels')})
        lm_loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        rul_mask = (task_ids == 2).to(device)
        if rul_mask.any():
            batch_input_ids = inputs['input_ids']
            B = batch_input_ids.size(0)
            adapter = model.get_input_embeddings().adapter
            adapter_device = next(adapter.parameters()).device if sum(1 for _ in adapter.parameters())>0 else device
            cnt = 0
            reg_losses = []
            for b in range(B):
                if not rul_mask[b]:
                    continue
                ids = batch_input_ids[b]
                mask = (ids >= signal_token_id)
                if mask.sum().item() != description_len:
                    continue
                sig_ids = ids[mask].unsqueeze(0).to(device)
                signal_feats = model.get_input_embeddings().signal_converter.get_signal(sig_ids, train_mode=model.training)
                signal_feats = signal_feats.to(adapter_device)
                with torch.set_grad_enabled(model.training):
                    rul_pred = adapter.predict_rul(signal_feats) 
                rul_pred = rul_pred.to(device)
                target = ruls[b]
                if target is None or (isinstance(target, float) and np.isnan(target)):
                    continue
                target = target.to(device)
                reg_losses.append(F.l1_loss(rul_pred, target.view_as(rul_pred)))
                cnt += 1
            if cnt > 0:
                reg_loss = torch.stack(reg_losses).mean()
        alpha = 1.0
        beta = 10.0
        total_loss = lm_loss * alpha + reg_loss * beta
        return (total_loss, outputs) if return_outputs else total_loss

def fine_tuning():
    hp = HyperParameters()
    model = get_bearllm()
    lora_config = LoraConfig(target_modules="all-linear",
                             task_type=TaskType.CAUSAL_LM,
                             r=hp.r,
                             lora_alpha=hp.lora_alpha,
                             lora_dropout=hp.lora_dropout)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    dataset = FineTuningDataset()
    train_args = TrainingArguments(
        output_dir='/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/lora_PHM2012',
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=hp.logging_steps,
        num_train_epochs=hp.num_train_epochs,
        save_steps=hp.save_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
    )
    trainer = MultiTaskTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collate_fn
    )
    trainer.train()
    trainer.model.save_pretrained('/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/lora_PHM2012')

if __name__ == "__main__":
    fine_tuning()
