"""
Unified runner for training and evaluating the new BERT-based model.
"""
import argparse

import torch
from sklearn.metrics import roc_auc_score

from config import config
from dataset import prepare_data
from model_new import create_new_model
from train import compute_metrics
from train_new_bert import BertTextMatchDataset, train as train_new_bert
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_new_bert(model_path, val_df, batch_size=32, num_workers=2):
    print("=" * 80)
    print("Evaluating new BERT model")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    bert = AutoModel.from_pretrained("bert-base-chinese")
    model = create_new_model(use_bert=True, bert_model=bert).to(config.DEVICE)

    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_dataset = BertTextMatchDataset(val_df, tokenizer, max_len=config.MAX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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

    auc = roc_auc_score(all_labels, all_probs)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print(f"\nAUC: {auc:.4f}")
    print(
        "Accuracy: {:.4f}  Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}".format(
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
    )
    return auc, metrics


def main():
    parser = argparse.ArgumentParser(description="Run new BERT-based model")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "eval", "all"],
        help="Run mode: train, eval, or all",
    )
    parser.add_argument("--model_path", type=str, default="best_model_new_bert.pth")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--freeze_bert", action="store_true", default=True)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_df, val_df = prepare_data()

    if args.mode in ["train", "all"]:
        train_new_bert(
            use_contrastive=args.contrastive,
            contrastive_weight=args.contrastive_weight,
            freeze_bert=args.freeze_bert,
            num_layers=args.num_layers,
            use_amp=not args.no_amp,
        )

    if args.mode in ["eval", "all"]:
        evaluate_new_bert(
            args.model_path,
            val_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
