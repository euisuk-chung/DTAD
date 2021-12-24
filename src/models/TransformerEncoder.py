import torch
import torch.nn as nn


class transformer_encoder(nn.Module):
    def __init__(self, seq_len, hidden_size, num_heads, ff_dim, dropout=0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.Attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ff_layer = nn.Sequential(
            nn.LayerNorm([seq_len, hidden_size], eps=1e-6),
            nn.Conv1d(seq_len, ff_dim, kernel_size=1),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, seq_len, kernel_size=1)
        )

    def forward(self, inputs):
        x = self.ln(inputs)
        x, x_weights = self.Attn(x, x, x)
        x = self.dropout(x)
        res = x + inputs
        x = self.ff_layer(res)
        return x + res


class MLP(nn.Module):
    def __init__(self, mlp_units, dropout, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.mlp_layers = []
        for i in range(len(mlp_units)-1):
            act = activation if i < len(mlp_units)-2 else output_activation
            if i!=len(mlp_units)-2:
                self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i+1]), nn.Dropout(dropout), act()]
            else:
                if act == None:
                    self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i + 1])]
                else:
                    self.mlp_layers += [nn.Linear(mlp_units[i], mlp_units[i+1]), act()]
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
    def forward(self, inputs):
        return self.mlp_layers(inputs)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            seq_len,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=0,
            mlp_dropout=0
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_size))
        self.num_transformer_blocks = num_transformer_blocks
        self.encoder = transformer_encoder(seq_len, hidden_size, num_heads, ff_dim, dropout)
        self.mlp = MLP(mlp_units, mlp_dropout)

    def forward(self, inputs):
        x = inputs + self.pos_embedding
        for _ in range(self.num_transformer_blocks):
            x = self.encoder(x)
        #         x = x.mean(dim=(1))
        output = self.mlp(x[:, -1])
        return output