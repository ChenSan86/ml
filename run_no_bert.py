"""
Unified runner for training and evaluating the new model without BERT.
"""
import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import config
from dataset import prepare_data, get_dataloaders
from model_new_no_bert import create_new_no_bert_model, NewTextMatchModelNoBert
from train import compute_metrics, plot_training_history, set_seed


def train(use_contrastive=False, contrastive_weight=0.2):
    print("=" * 80)
    print("🚀 训练新模型 (无BERT)")
    print("=" * 80)

    set_seed(config.SEED)
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(
        train_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS
    )

    model = create_new_no_bert_model().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
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
                cont_loss = NewTextMatchModelNoBert.contrastive_loss(repr1, repr2, labels)
            else:
                logits = model(query1, query2)
                cont_loss = 0.0

            bce_loss = criterion(logits, labels)
            loss = bce_loss + contrastive_weight * cont_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_auc = compute_metrics(
            all_labels, (torch.tensor(all_probs) > 0.5).int(), all_probs
        )["auc"]

        model.eval()
        val_total_loss = 0.0
        val_labels, val_probs, val_preds = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                query1 = batch["query1"].to(config.DEVICE)
                query2 = batch["query2"].to(config.DEVICE)
                labels = batch["label"].to(config.DEVICE)

                logits = model(query1, query2)
                loss = criterion(logits, labels)

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
                {
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                },
                "best_model_new_no_bert.pth",
            )

    plot_training_history(history, "training_history_new_no_bert.png")
    return model, history


@torch.no_grad()
def evaluate(model_path, val_df):
    print("=" * 80)
    print("Evaluating new model (no BERT)")
    print("=" * 80)

    model = create_new_no_bert_model().to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_loader = get_dataloaders(
        val_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS
    )

    all_labels, all_probs, all_preds = [], [], []
    for batch in val_loader:
        query1 = batch["query1"].to(config.DEVICE)
        query2 = batch["query2"].to(config.DEVICE)
        labels = batch["label"].cpu().numpy()

        logits = model(query1, query2)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_labels.extend(labels)
        all_probs.extend(probs)
        all_preds.extend(preds)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print(
        "Accuracy: {:.4f}  Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}  AUC: {:.4f}".format(
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc"],
        )
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run new model without BERT")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "eval", "all"],
        help="Run mode: train, eval, or all",
    )
    parser.add_argument("--model_path", type=str, default="best_model_new_no_bert.pth")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    args = parser.parse_args()

    train_df, val_df = prepare_data()

    if args.mode in ["train", "all"]:
        train(
            use_contrastive=args.contrastive,
            contrastive_weight=args.contrastive_weight,
        )

    if args.mode in ["eval", "all"]:
        evaluate(args.model_path, val_df)


if __name__ == "__main__":
    main()
