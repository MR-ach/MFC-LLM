import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tqdm import tqdm
from datasets import get_loaders  
from F2LNet import F2LNet   

F2LNet_weights = '/home/hzm/MFC-LLM/LLM/F2LNet_LLM_weight/F2LNet_PHM2012'  

class HyperParameters:
    def __init__(self):
        self.batch_size = 1024
        self.num_workers = 10
        self.lr = 1e-4
        self.lr_patience = 150
        self.lr_factor = 0.5
        self.epoch_max = 50
        self.device = 'cuda'
        self.lambda_rul = 1  

class PreTrainer:
    def __init__(self):
        self.hp = HyperParameters()
        train_loader, val_loader, test_loader = get_loaders(self.hp.batch_size, self.hp.num_workers)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = F2LNet().to(self.hp.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=self.hp.lr_patience,
                                                                    factor=self.hp.lr_factor)
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.rul_criterion = torch.nn.MSELoss()
        self.best_val_loss = 1e10
        self.best_val_acc = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0  
        for data, cls_label, rul_label in tqdm(self.train_loader):                              
            data = data.to(self.hp.device)
            cls_label = cls_label.to(self.hp.device)
            rul_label = rul_label.to(self.hp.device).float().unsqueeze(1)  # [B,1]
            self.optimizer.zero_grad()
            cls_out, life_out = self.model(data)
            loss_cls = self.cls_criterion(cls_out, cls_label)
            loss_rul = self.rul_criterion(life_out, rul_label)
            loss = loss_cls + self.hp.lambda_rul * loss_rul
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            for data, cls_label, rul_label in self.val_loader:
                data = data.to(self.hp.device)
                cls_label = cls_label.to(self.hp.device)
                rul_label = rul_label.to(self.hp.device).float().unsqueeze(1)
                cls_out, life_out = self.model(data)
                loss_cls = self.cls_criterion(cls_out, cls_label)
                loss_rul = self.rul_criterion(life_out, rul_label)
                loss = loss_cls + self.hp.lambda_rul * loss_rul
                val_loss += loss.item()
                correct += (cls_out.argmax(1) == cls_label).sum().item()
            val_loss /= len(self.val_loader)
            val_acc = correct / len(self.val_loader.dataset)           
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        return val_loss, val_acc

    def test_epoch(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_rul_preds = []
        all_rul_labels = []
        correct = 0
        mse_total = 0
        with torch.no_grad():
            for data, cls_label, rul_label in self.test_loader:
                data = data.to(self.hp.device)
                cls_label = cls_label.to(self.hp.device).view(-1).long() 
                rul_label = rul_label.to(self.hp.device).float().unsqueeze(1)
                cls_out, life_out = self.model(data)
                preds = cls_out.argmax(1)
                correct += (preds == cls_label).sum().item()
                mse_total += self.rul_criterion(life_out, rul_label).item() * data.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cls_label.cpu().numpy())
                all_rul_preds.extend(life_out.cpu().numpy())
                all_rul_labels.extend(rul_label.cpu().numpy())
        test_acc = correct / len(self.test_loader.dataset)
        test_rul_mse = mse_total / len(self.test_loader.dataset)
        return test_acc, test_rul_mse

    def train(self):
        patience_limit = 7  
        patience_counter = 0
        for epoch in range(self.hp.epoch_max):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_weights(F2LNet_weights)
                patience_counter = 0 
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        test_acc, test_rul_mse = self.test_epoch()
        print(f'Test Acc: {test_acc:.4f}, Test RUL MSE: {test_rul_mse:.4f}')

if __name__ == "__main__":
    trainer = PreTrainer()
    trainer.train()
