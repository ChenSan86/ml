"""
Train the new model with bert-base-chinese encoder.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import config
from dataset import prepare_data
from model_new import create_new_model, NewTextMatchModel
from train import set_seed, compute_metrics, plot_training_history


class BertTextMatchDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.query1 = df["query1"].values
        self.query2 = df["query2"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode(self, text):
        encoded = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        q1_ids, q1_mask = self._encode(self.query1[idx])
        q2_ids, q2_mask = self._encode(self.query2[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            "query1": q1_ids,
            "query2": q2_ids,
            "mask1": q1_mask,
            "mask2": q2_mask,
            "label": label,
        }


def get_bert_dataloaders(train_df, val_df, tokenizer, batch_size=32, num_workers=2):
    train_dataset = BertTextMatchDataset(train_df, tokenizer, max_len=config.MAX_LEN)
    val_dataset = BertTextMatchDataset(val_df, tokenizer, max_len=config.MAX_LEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train(use_contrastive=False, contrastive_weight=0.2, freeze_bert=False):
    print("=" * 80)
    print("🚀 训练新模型 (bert-base-chinese)")
    print("=" * 80)

    set_seed(config.SEED)
    train_df, val_df = prepare_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    bert = AutoModel.from_pretrained("bert-base-chinese")

    model = create_new_model(use_bert=True, bert_model=bert).to(config.DEVICE)
    if freeze_bert:
        for param in model.bert.bert_model.parameters():
            param.requires_grad = False

    train_loader, val_loader = get_bert_dataloaders(
        train_df, val_df, tokenizer, batch_size=32, num_workers=2
    )

    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "val_metrics": [],
    }
    best_auc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        start_time = time.time()

        model.train()
        total_loss = 0.0
        all_labels, all_probs = [], []

        for batch in tqdm(train_loader, desc="Training"):
            query1 = batch["query1"].to(config.DEVICE)
            query2 = batch["query2"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            optimizer.zero_grad()

            if use_contrastive:
                logits, repr1, repr2 = model(query1, query2, return_reprs=True)
                cont_loss = NewTextMatchModel.contrastive_loss(repr1, repr2, labels)
            else:
                logits = model(query1, query2)
                cont_loss = 0.0

            bce_loss = bce_criterion(logits, labels)
            loss = bce_loss + contrastive_weight * cont_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_auc = 0.0
        if len(set(all_labels)) > 1:
            train_auc = compute_metrics(all_labels, (torch.tensor(all_probs) > 0.5).int(), all_probs)[
                "auc"
            ]

        model.eval()
        val_total_loss = 0.0
        val_labels, val_probs, val_preds = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                query1 = batch["query1"].to(config.DEVICE)
                query2 = batch["query2"].to(config.DEVICE)
                labels = batch["label"].to(config.DEVICE)

                logits = model(query1, query2)
                loss = bce_criterion(logits, labels)

                val_total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs > 0.5).astype(int))
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_total_loss / len(val_loader)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - start_time
        print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, AUC={val_metrics['auc']:.4f}")
        print(f"Time:  {elapsed:.1f}s")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {"model_state_dict": model.state_dict(), "best_auc": best_auc},
                "best_model_new_bert.pth",
            )

    plot_training_history(history, "training_history_new_bert.png")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train new model with bert-base-chinese")
    parser.add_argument("--contrastive", action="store_true", help="enable contrastive loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--freeze_bert", action="store_true")
    args = parser.parse_args()

    train(
        use_contrastive=args.contrastive,
        contrastive_weight=args.contrastive_weight,
        freeze_bert=args.freeze_bert,
    )


if __name__ == "__main__":
    main()
