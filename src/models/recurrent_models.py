from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class RecurrentClassifier(nn.Module):
    """Deep Learning Recurrent Classifier supporting both LSTM (Algorithm 3) and BiLSTM (Algorithm 4)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token ID tensor of shape (batch_size, seq_len).
            lengths: Optional sequence length tensor.

        Returns:
            Logits tensor of shape (batch_size,).
        """
        # Shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(x))

        # Pass through LSTM
        # out shape: (batch_size, seq_len, hidden_dim * num_directions)
        out, (hn, cn) = self.lstm(embedded)

        # Global max or average pooling across sequence dimension to capture salient distress phrases
        # Mask out padding tokens
        mask = (x != self.embedding.padding_idx).unsqueeze(-1).float()
        out_masked = out * mask
        sum_masked = torch.sum(out_masked, dim=1)
        lengths_clamped = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = sum_masked / lengths_clamped

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes calibrated probability of positive class (Distress language)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


def build_recurrent_model(
    model_type: str,
    vocab_size: int,
    config: Optional[Dict[str, Any]] = None,
    pad_idx: int = 0,
) -> RecurrentClassifier:
    """Factory function to build LSTM or BiLSTM from configuration.

    Args:
        model_type: 'lstm' or 'bilstm'.
        vocab_size: Size of vocabulary.
        config: Configuration dictionary.
        pad_idx: Padding index.

    Returns:
        Instantiated RecurrentClassifier.
    """
    cfg_key = "bilstm" if model_type.lower() == "bilstm" else "lstm"
    cfg = config.get("models", {}).get(cfg_key, {}) if config else {}

    is_bidirectional = cfg.get("bidirectional", True if model_type.lower() == "bilstm" else False)

    return RecurrentClassifier(
        vocab_size=vocab_size,
        embedding_dim=cfg.get("embedding_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        bidirectional=is_bidirectional,
        pad_idx=pad_idx,
    )
