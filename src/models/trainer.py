from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger("recurrent_trainer")


class RecurrentModelTrainer:
    """Trainer for LSTM and BiLSTM models with validation and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        logger.info(f"Initialized RecurrentModelTrainer on device: {self.device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y, lengths in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x, lengths)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)

        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluates model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y, lengths in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x, lengths)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_y)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = {
            "accuracy": float(accuracy_score(all_targets, all_preds)),
            "f1": float(f1_score(all_targets, all_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(all_targets, all_probs)) if len(set(all_targets)) > 1 else 0.0,
        }
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        patience: int = 3,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Fits model with early stopping based on validation loss."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs (patience={patience})...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_metrics["f1"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_path)
                    logger.info(f"--> Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"Loaded best weights from {checkpoint_path}")

        return history

    def get_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (probabilities, predictions, targets) for dataloader."""
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y, lengths in dataloader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x, lengths)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(batch_y.numpy())

        probs_arr = np.array(all_probs)
        preds_arr = (probs_arr >= 0.5).astype(int)
        return probs_arr, preds_arr, np.array(all_targets)
