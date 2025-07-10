import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, hidden_size=768, final_dropout=0.1, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(final_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x