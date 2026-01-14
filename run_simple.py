"""
Unified runner for training and evaluating the simplified model.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import prepare_data, TextMatchDataset
from mode_simple import create_simple_model
from train import compute_metrics, plot_training_history, set_seed


def build_dataloaders(
    train_df,
    val_df,
    batch_size,
    num_workers,
    prefetch_factor,
    persistent_workers,
    pin_memory,
):
    train_dataset = TextMatchDataset(train_df, max_len=config.MAX_LEN)
    val_dataset = TextMatchDataset(val_df, max_len=config.MAX_LEN)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = prefetch_factor
        common_kwargs["persistent_workers"] = persistent_workers
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader


def train(
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    cudnn_benchmark=True,
):
    print("=" * 80)
    print("🚀 训练简化模型")
    print("=" * 80)

    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    train_df, val_df = prepare_data()
    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    model = create_simple_model().to(config.DEVICE)
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

            logits = model(query1, query2)
            loss = criterion(logits, labels)
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
                "best_model_simple.pth",
            )

    plot_training_history(history, "training_history_simple.png")
    return model, history


@torch.no_grad()
def evaluate(model_path, val_df, batch_size=256, num_workers=4, pin_memory=True):
    print("=" * 80)
    print("Evaluating simplified model")
    print("=" * 80)

    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    state = checkpoint["model_state_dict"]

    model = create_simple_model().to(config.DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"⚠️  忽略未匹配参数: {unexpected}")
    if missing:
        print(f"⚠️  缺失参数: {missing}")
    model.eval()

    val_dataset = TextMatchDataset(val_df, max_len=config.MAX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    parser = argparse.ArgumentParser(description="Run simplified model")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "eval", "all"],
        help="Run mode: train, eval, or all",
    )
    parser.add_argument("--model_path", type=str, default="best_model_simple.pth")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", action="store_true")
    args = parser.parse_args()

    train_df, val_df = prepare_data()

    if args.mode in ["train", "all"]:
        train(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
            cudnn_benchmark=not args.no_cudnn_benchmark,
        )

    if args.mode in ["eval", "all"]:
        evaluate(
            args.model_path,
            val_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )


if __name__ == "__main__":
    main()
