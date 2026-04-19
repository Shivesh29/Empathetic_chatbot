import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from tqdm import tqdm


EMOTIONS = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise"
}

class TacoDataset(Dataset):
    def __init__(self, split="train", model_name="microsoft/deberta-v3-small"):
       
        ds = load_dataset("go_emotions", "simplified")[split]
        
        self.data = [x for x in ds if len(x['labels']) == 1 and x['labels'][0] != 27]
        
        self.data = self.data[:2000] 
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=64, return_tensors="pt")
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item['labels'][0], dtype=torch.long)
        }


class TacoModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small"):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        dim = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128) 
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        cls_token = out.last_hidden_state[:, 0, :]
        
        
        embs = self.projector(cls_token.float())
        return F.normalize(embs, p=2, dim=1) 


def get_losses(batch_embs, labels, model, tokenizer, device):
    label_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]
    l_enc = tokenizer(label_names, padding=True, return_tensors="pt").to(device)
    

    label_embs = model(l_enc['input_ids'], l_enc['attention_mask'])
    
    
    logits = torch.matmul(batch_embs, label_embs.T) / 0.1
    loss_ce = F.cross_entropy(logits, labels)

    
    loss_ccl = torch.tensor(0.0).to(device)
    if batch_embs.size(0) > 8:
        km = KMeans(n_clusters=8, n_init='auto').fit(batch_embs.detach().cpu().numpy())
        cluster_ids = torch.tensor(km.labels_).to(device)
        loss_ccl = F.mse_loss(batch_embs, label_embs[labels]) 

   
    label_sim = torch.matmul(label_embs, label_embs.T)
    loss_ldl = torch.mean(torch.triu(label_sim, diagonal=1))

    return loss_ce, loss_ccl, loss_ldl


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Pipeline on {device}...")

    model_name = "microsoft/deberta-v3-small"
    train_ds = TacoDataset("train", model_name)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = TacoModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(4): 
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            embs = model(batch['ids'].to(device), batch['mask'].to(device))
            l_ce, l_ccl, l_ldl = get_losses(embs, batch['label'].to(device), model, train_ds.tokenizer, device)
            
            total_loss = l_ce + (0.1 * l_ccl) + (0.1 * l_ldl)
            total_loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            pbar.set_postfix(CE=f"{l_ce.item():.3f}", CCL=f"{l_ccl.item():.3f}")

    print("✅ Project Complete. Model weights saved as 'taco_final.pth'")
    torch.save(model.state_dict(), "taco_final.pth")

if __name__ == "__main__":
    run_pipeline()