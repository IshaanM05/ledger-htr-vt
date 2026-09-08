import torch


class CTCCodec:
    """Character<->index mapping for CTC training and greedy decoding.
    Index 0 is reserved for the CTC blank symbol."""

    def __init__(self, alphabet: list[str]):
        self.blank_index = 0
        self.itos = ["<blank>"] + list(alphabet)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def num_classes(self) -> int:
        return len(self.itos)

    def encode(self, texts: list[str]) -> tuple[torch.IntTensor, torch.IntTensor]:
        """Flattened target indices + per-sample lengths, as torch.nn.CTCLoss expects."""
        lengths = torch.IntTensor([len(t) for t in texts])
        flat = torch.IntTensor([self.stoi[ch] for text in texts for ch in text])
        return flat, lengths

    def decode_greedy(self, index_batch: torch.Tensor) -> list[str]:
        """index_batch: (B, T) argmax class index per timestep. Collapses
        repeated consecutive symbols and drops blanks, per standard CTC
        decoding (Graves et al., 2006)."""
        texts = []
        for row in index_batch.tolist():
            chars = []
            prev = -1
            for idx in row:
                if idx != self.blank_index and idx != prev:
                    chars.append(self.itos[idx])
                prev = idx
            texts.append("".join(chars))
        return texts
