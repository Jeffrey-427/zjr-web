import math
import random
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Config
TRAIN_PATH = "train.csv"
TEST_PATH = "test2.csv"

MAX_LEN = 100
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

BATCH_SIZE = 256
EPOCHS = 8
LR = 5e-4
WEIGHT_DECAY = 1e-6
WARMUP_EPOCHS = 5
PATIENCE = 10
ONLY_PREDICT = False   # 设为 True：只生成 submission，不训练lo


# Popularity rerank strength
ALPHA_POP = 0

# Data
def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df["history"] = train_df["history_item_id"].apply(ast.literal_eval)
    test_df["history"] = test_df["history_item_id"].apply(ast.literal_eval)

    all_items = set()
    for h in train_df["history"]:
        all_items.update(h)
    all_items.update(train_df["item_id"].tolist())
    for h in test_df["history"]:
        all_items.update(h)

    sorted_items = sorted(list(all_items))
    item2idx = {item: idx + 1 for idx, item in enumerate(sorted_items)}
    idx2item = {idx + 1: item for idx, item in enumerate(sorted_items)}
    num_items = len(sorted_items) + 1  # + PAD=0

    print(f"  Train rows: {len(train_df)} | Items: {num_items-1} (+PAD)")
    return train_df, test_df, item2idx, idx2item, num_items


def build_popularity(train_df: pd.DataFrame, item2idx: dict, num_items: int) -> np.ndarray:
    pop = np.zeros(num_items, dtype=np.float32)
    for h in train_df["history"]:
        for it in h:
            pop[item2idx.get(it, 0)] += 1.0
    for it in train_df["item_id"].tolist():
        pop[item2idx.get(it, 0)] += 1.0

    pop = np.log1p(pop)
    if pop.max() > 0:
        pop = pop / (pop.max() + 1e-8)
    return pop


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, item2idx: dict, max_len: int):
        self.max_len = max_len
        self.item2idx = item2idx

        self.user_ids = df["user_id"].tolist()
        self.hists = df["history"].tolist()
        self.targets = df["item_id"].tolist()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        hist = [self.item2idx.get(x, 0) for x in self.hists[idx]]
        target = self.item2idx.get(self.targets[idx], 0)

        hist = hist[-self.max_len:]
        seq = [0] * (self.max_len - len(hist)) + hist  # left pad
        labels = seq[1:] + [target]  # last position predicts target

        return {
            "user_id": int(self.user_ids[idx]),
            "seq": torch.LongTensor(seq),
            "labels": torch.LongTensor(labels),
            "target": int(target),
        }


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, item2idx: dict, max_len: int):
        self.max_len = max_len
        self.item2idx = item2idx
        self.user_ids = df["user_id"].tolist()
        self.hists = df["history"].tolist()
        self.targets = df["item_id"].tolist()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        hist = [self.item2idx.get(x, 0) for x in self.hists[idx]]
        target = self.item2idx.get(self.targets[idx], 0)

        hist = hist[-self.max_len:]
        seq = [0] * (self.max_len - len(hist)) + hist

        return {
            "user_id": int(self.user_ids[idx]),
            "seq": torch.LongTensor(seq),
            "target": int(target),
        }


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, item2idx: dict, max_len: int):
        self.max_len = max_len
        self.item2idx = item2idx
        self.user_ids = df["user_id"].tolist()
        self.hists = df["history"].tolist()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        hist = [self.item2idx.get(x, 0) for x in self.hists[idx]]
        hist = hist[-self.max_len:]
        seq = [0] * (self.max_len - len(hist)) + hist
        return {
            "user_id": int(self.user_ids[idx]),
            "seq": torch.LongTensor(seq),
        }

# Model
class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, key_padding_mask: torch.Tensor):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,L,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,L,hd]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]

        # Causal mask: prevent attending to future
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask: prevent attending to PAD keys
        attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                    # [B,H,L,hd]
        out = out.transpose(1, 2).reshape(B, L, D)     # [B,L,D]
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = SimpleSelfAttention(hidden_size, num_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, key_padding_mask: torch.Tensor):
        x = x + self.attn(self.ln1(x), causal_mask, key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items: int, hidden_size: int, max_len: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, num_items)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        device = seq.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()  # [L,L]
        key_padding_mask = (seq == 0)  # [B,L]

        pos = torch.arange(L, device=device).unsqueeze(0)  # [1,L]
        x = self.item_emb(seq) + self.pos_emb(pos)

        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)

        x = self.ln(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.forward_hidden(seq)        # [B,L,D]
        out = h[:, -1, :]                   # last position represents last history item (due to training alignment)
        scores = self.out(out)              # [B,num_items]
        return scores

# LR schedule
def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr_ratio: float = 0.05) -> float:
    """Warmup + cosine with min_lr_ratio (no cliff drop)."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


# Train / Eval / Predict
def train_epoch(model: SASRec, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        seq = batch["seq"].to(device)          # [B,L]
        labels = batch["labels"].to(device)    # [B,L]

        optimizer.zero_grad()
        h = model.forward_hidden(seq)          # [B,L,D]
        logits = model.out(h)                  # [B,L,num_items]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0
        )

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)


def apply_seen_filter(scores: torch.Tensor, seq: torch.Tensor, mode: str = "full"):
    mode = (mode or "full").lower()

    # always filter PAD=0
    scores[:, 0] = -1e9

    if mode == "pad":
        return

    if mode == "last":
        last = seq[:, -1]  # [B]
        mask = last != 0
        if mask.any():
            rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
            cols = last[mask]
            scores[rows, cols] = -1e9
        return

    if mode == "full":
        scores.scatter_(1, seq, -1e9)
        scores[:, 0] = -1e9
        return

    raise ValueError(f"Unknown filter mode: {mode}")



def evaluate(model, loader, device, pop_t, alpha_pop,
             k=10, restore_target=True, filter_mode="full"):
    model.eval()
    mrr, hits, total = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            seq = batch["seq"].to(device)                              # [B,L]
            targets = torch.as_tensor(batch["target"], device=device).long()
            # [B]

            scores = model(seq)                                        # [B,num_items]
            if alpha_pop > 0:
                scores = scores + alpha_pop * pop_t.unsqueeze(0)

            if torch.isnan(scores).any():
                continue

            # filter seen items (same as predict)
            if restore_target:
                # old behavior: optimistic eval (won't be hurt if target is in history)
                t_score = scores.gather(1, targets.unsqueeze(1))  # [B,1]
                apply_seen_filter(scores, seq)
                scores.scatter_(1, targets.unsqueeze(1), t_score)
            else:
                # submission-style eval: no "cheat restore", exactly like predict()
                apply_seen_filter(scores, seq, mode=filter_mode)

            _, topk = torch.topk(scores, k, dim=-1)                    # [B,k]
            topk = topk.cpu().numpy()

            t_cpu = targets.cpu().numpy()
            for i, t in enumerate(t_cpu):
                if t in topk[i]:
                    rank = int(np.where(topk[i] == t)[0][0]) + 1
                    mrr += 1.0 / rank
                    hits += 1.0
                total += 1

    if total == 0:
        return 0.0, 0.0
    return mrr / total, hits / total


def predict(model: SASRec, loader: DataLoader, idx2item: dict, device: torch.device, pop_t: torch.Tensor, alpha_pop: float, k: int = 10):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            seq = batch["seq"].to(device)
            user_ids = batch["user_id"]

            scores = model(seq)
            if alpha_pop > 0:
                scores = scores + alpha_pop * pop_t.unsqueeze(0)

            # filter seen items for recommendation
            apply_seen_filter(scores, seq, mode="pad")
            _, topk = torch.topk(scores, k, dim=-1)
            topk = topk.cpu().numpy()

            for i in range(len(user_ids)):
                uid = int(user_ids[i])
                items = [int(idx2item[idx]) for idx in topk[i] if idx in idx2item]
                results.append({"user_id": uid, "item_id": items})

    return pd.DataFrame(results)


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("SASRec Model")
    print("=" * 60)

    train_df, test_df, item2idx, idx2item, num_items = load_data()
    pop = build_popularity(train_df, item2idx, num_items)
    pop_t = torch.tensor(pop, device=device)

    # split (random 90/10)
    n = len(train_df)
    indices = np.random.permutation(n)
    n_val = int(n * 0.1)
    val_data = train_df.iloc[indices[:n_val]].reset_index(drop=True)
    tr_data = train_df.iloc[indices[n_val:]].reset_index(drop=True)
    print(f"Train: {len(tr_data)} | Val: {len(val_data)}")

    train_ds = TrainDataset(tr_data, item2idx, MAX_LEN)
    val_ds = EvalDataset(val_data, item2idx, MAX_LEN)
    test_ds = TestDataset(test_df, item2idx, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SASRec(num_items, HIDDEN_SIZE, MAX_LEN, NUM_LAYERS, NUM_HEADS, DROPOUT).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_mrr = 0.0
    no_improve = 0

    print("\nTraining...")
    print("\nTraining...")
    if not ONLY_PREDICT:
        for epoch in range(1, EPOCHS + 1):
            lr = get_lr(epoch - 1, WARMUP_EPOCHS, EPOCHS, LR, min_lr_ratio=0.05)
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss = train_epoch(model, train_loader, optimizer, device)
            mrr_opt, hit_opt = evaluate(
                model, val_loader, device, pop_t, ALPHA_POP, k=10, restore_target=True
            )
            mrr_sub, hit_sub = evaluate(
                model, val_loader, device, pop_t, ALPHA_POP, k=10, restore_target=False
            )

            print(
                f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                f"MRR@10(opt) {mrr_opt:.4f} Hit@10(opt) {hit_opt:.4f} | "
                f"MRR@10(sub) {mrr_sub:.4f} Hit@10(sub) {hit_sub:.4f} | LR {lr:.6f}"
            )

            if mrr_sub > best_mrr:
                best_mrr = mrr_sub
                torch.save(model.state_dict(), "best.pt")
                print(f"  -> Best updated: {best_mrr:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print("Early stop.")
                    break
    else:
        print("ONLY_PREDICT=True: skip training.")

    # Validate submission-style MRR with different seen-filter modes
    print("\n[Check] VAL submission-style MRR with different filter modes")

    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.eval()

    for fm in ["full", "pad", "last"]:
        mrr_sub, hit_sub = evaluate(
            model,
            val_loader,
            device,
            pop_t,
            ALPHA_POP,
            k=10,
            restore_target=False,  # submission-style
            filter_mode=fm
        )
        print(f"[VAL SUB] mode={fm:<4s}  MRR@10={mrr_sub:.4f}  Hit@10={hit_sub:.4f}")

    # Predict
    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.eval()

    result = predict(
        model,
        test_loader,
        idx2item,
        device,
        pop_t,
        ALPHA_POP,
        k=10
    )
    # Ensure item_id column is a string like "[a, b, c, ...]"
    result["item_id"] = result["item_id"].apply(
        lambda x: "[" + ", ".join(map(str, x)) + "]"
    )

    # 再保存 CSV
    result.to_csv("submission.csv", index=False)

    print("\nOutput sample:")
    print(result.head(3))
    print("\nDone. Saved: best.pt, submission.csv")


if __name__ == "__main__":
    main()
