from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset


class Vocabulary:
    """Builds and manages vocabulary mapping words to indices."""

    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}
        self.pad_idx: int = 0
        self.unk_idx: int = 1

    def build_vocab(self, texts: List[str], max_vocab_size: int = 15000, min_freq: int = 2) -> None:
        """Constructs vocabulary from a list of input texts.

        Args:
            texts: List of text strings.
            max_vocab_size: Maximum number of tokens in vocabulary.
            min_freq: Minimum token occurrence count.
        """
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        # Filter and sort tokens by frequency
        most_common = counter.most_common(max_vocab_size - len(self.word2idx))
        for word, count in most_common:
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_seq_len: int = 128) -> List[int]:
        """Encodes text to a list of token IDs with truncation and padding.

        Args:
            text: Input string.
            max_seq_len: Target sequence length.

        Returns:
            List of integers of length `max_seq_len`.
        """
        tokens = text.split()[:max_seq_len]
        indices = [self.word2idx.get(w, self.unk_idx) for w in tokens]
        if len(indices) < max_seq_len:
            indices += [self.pad_idx] * (max_seq_len - len(indices))
        return indices

    def decode(self, indices: List[int]) -> str:
        """Decodes token IDs back to a string."""
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices if idx != self.pad_idx]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_seq_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        token_ids = self.vocab.encode(text, self.max_seq_len)
        length = min(len(text.split()), self.max_seq_len)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
            length,
        )
