import torch
import torch.nn as nn
import transformers

class GPT(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            max_len,
            **kwargs
    ):
        super().__init__()
        self.input_dim = input_size
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = transformers.GPT2Model(config)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))
        self.embed_token = nn.Linear(self.input_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict = nn.Linear(hidden_size, self.input_dim)

    def forward(self, token, attention_mask=None):
        batch_size, seq_length = token.shape[0], token.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(self.transformer.device)

        # embed each modality with a different head
        token_embeddings = self.embed_token(token)

        # time embeddings are treated similar to positional embeddings
        token_embeddings = token_embeddings + self.pos_embedding

        # which works nice in an autoregressive sense since states predict actions
        inputs = self.embed_ln(token_embeddings)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # get predictions
        preds = self.predict(x)

        return preds[:, -1]