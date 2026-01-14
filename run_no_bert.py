"""
Unified runner for training and evaluating the new model without BERT.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import TextMatchDataset, load_data
from model_new_no_bert import create_new_no_bert_model, NewTextMatchModelNoBert
from train import compute_metrics, plot_training_history, set_seed
from sklearn.model_selection import train_test_split


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


def split_data_8_1_1():
    files = [config.TRAIN_FILE_1, getattr(config, "TRAIN_FILE_2", None)]
    files = [f for f in files if f]
    df = load_data(files)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config.SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.SEED,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def train(
    train_df,
    val_df,
    contrastive_weight=0.2,
    batch_size=256,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    cudnn_benchmark=True,
    use_pos_weight=True,
):
    print("=" * 80)
    print("🚀 训练新模型 (无BERT)")
    print("=" * 80)

    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    model = create_new_no_bert_model().to(config.DEVICE)
    if use_pos_weight:
        pos_count = (train_df["label"] == 1).sum()
        neg_count = (train_df["label"] == 0).sum()
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=config.DEVICE)
        print(f"⚖️  使用 pos_weight={pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
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

            logits, repr1, repr2 = model(query1, query2, return_reprs=True)
            cont_loss = NewTextMatchModelNoBert.contrastive_loss(repr1, repr2, labels)

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
def evaluate(
    model_path,
    val_df,
    test_df,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    threshold_metric="f1",
    threshold_step=0.01,
):
    print("=" * 80)
    print("Evaluating new model (no BERT)")
    print("=" * 80)

    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    state = checkpoint["model_state_dict"]

    # Infer NUM_LAYERS from fusion.proj.weight shape: [hidden_dim, num_layers*hidden_dim*3]
    proj_weight = state.get("fusion.proj.weight")
    if proj_weight is not None:
        hidden_dim = proj_weight.shape[0]
        concat_dim = proj_weight.shape[1]
        inferred_layers = concat_dim // (hidden_dim * 3)
        if inferred_layers > 0 and inferred_layers != config.NUM_LAYERS:
            print(
                f"⚙️  从权重推断 NUM_LAYERS={inferred_layers} "
                f"(当前={config.NUM_LAYERS})，将临时覆盖以匹配权重"
            )
            config.NUM_LAYERS = inferred_layers

    model = create_new_no_bert_model().to(config.DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"⚠️  忽略未匹配参数: {unexpected}")
    model.eval()

    def predict_probs(dataframe):
        dataset = TextMatchDataset(dataframe, max_len=config.MAX_LEN)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        all_labels, all_probs = [], []
        for batch in loader:
            query1 = batch["query1"].to(config.DEVICE)
            query2 = batch["query2"].to(config.DEVICE)
            labels = batch["label"].cpu().numpy()

            logits = model(query1, query2)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)
        return np.array(all_labels), np.array(all_probs)

    val_labels, val_probs = predict_probs(val_df)
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    thresholds = np.arange(threshold_step, 1.0, threshold_step)
    for thr in thresholds:
        preds = (val_probs >= thr).astype(int)
        metrics = compute_metrics(val_labels, preds, val_probs)
        score = metrics.get(threshold_metric, 0.0)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_metrics(
            val_labels, (val_probs >= 0.5).astype(int), val_probs
        )

    print(
        f"🔧 验证集最优阈值: {best_threshold:.2f} (metric={threshold_metric}) "
        f"Precision={best_metrics['precision']:.4f} Recall={best_metrics['recall']:.4f} "
        f"F1={best_metrics['f1']:.4f} AUC={best_metrics['auc']:.4f}"
    )

    test_labels, test_probs = predict_probs(test_df)
    test_preds = (test_probs >= best_threshold).astype(int)
    metrics = compute_metrics(test_labels, test_preds, test_probs)
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
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_cudnn_benchmark", action="store_true")
    parser.add_argument("--no_pos_weight", action="store_true", default=True)
    parser.add_argument("--threshold_metric", type=str, default="f1")
    parser.add_argument("--threshold_step", type=float, default=0.01)
    args = parser.parse_args()

    train_df, val_df, test_df = split_data_8_1_1()

    if args.mode in ["train", "all"]:
        train(
            train_df,
            val_df,
            contrastive_weight=args.contrastive_weight,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
            cudnn_benchmark=not args.no_cudnn_benchmark,
            use_pos_weight=not args.no_pos_weight,
        )

    if args.mode in ["eval", "all"]:
        evaluate(
            args.model_path,
            val_df,
            test_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            threshold_metric=args.threshold_metric,
            threshold_step=args.threshold_step,
        )


if __name__ == "__main__":
    main()
