import torch
import torch.nn as nn

from torch.nn import functional as F

from . import hyper_params, data_sets, data_tool
from . import make_batch, estimate_loss


class SingleSelfAttention(nn.Module):
    def __init__(self, ndim, embeddings_ndim, embeddings_size):
        super().__init__()

        self.Wk = nn.Linear(embeddings_ndim, ndim, bias=False)
        self.Wq = nn.Linear(embeddings_ndim, ndim, bias=False)
        self.Wv = nn.Linear(embeddings_ndim, ndim, bias=False)

        self.dropout = nn.Dropout(hyper_params.dropout)

        mask = torch.tril(torch.ones(embeddings_size, embeddings_size))
        self.register_buffer('self_attention_mask', mask)


    def forward(self, Z):
        _, T, C = Z.shape # B, T, C
        K = self.Wk(Z) # B, T, ndim
        Q = self.Wq(Z) # B, T, ndim
        attention_scores = Q @ K.transpose(-2, -1)
        attention_weights = attention_scores / (C ** -0.5)
        self_attention_weights = attention_weights.masked_fill(
            self.self_attention_mask[:T, :T] == 0,
            float('-inf')
        )
        self_attention_probs = F.softmax(self_attention_weights, dim=1)
        self.dropout(self_attention_probs)
        V = self.Wv(Z)
        return self_attention_probs @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, head_ndim, heads_count, embeddings_ndim, embeddings_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleSelfAttention(head_ndim, embeddings_ndim, embeddings_size)
             for _ in range(heads_count)]
        )
        self.projs = nn.Linear(embeddings_ndim, embeddings_ndim)
        self.dropout = nn.Dropout(hyper_params.dropout)

    def forward(self, X):
        Z = torch.cat([head(X) for head in self.heads], dim=-1)
        Z = self.dropout(self.projs(Z))
        return Z


class FeedForwardNetwork(nn.Module):
    def __init__(self, embeddings_ndim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embeddings_ndim, 4 * embeddings_ndim),
            nn.ReLU(),
            nn.Linear(4 * embeddings_ndim, embeddings_ndim),
            nn.Dropout(hyper_params.dropout)
        )

    def forward(self, X):
        return self.net(X)


class SelfAttentionBlock(nn.Module):
    def __init__(self, heads_count, embeddings_ndim, embeddings_size):
        super().__init__()
        head_ndim = embeddings_ndim // heads_count
        self._1st_ln = nn.LayerNorm(embeddings_ndim)
        self._mha = MultiHeadAttention(
            heads_count=heads_count,
            head_ndim=head_ndim,
            embeddings_ndim=embeddings_ndim,
            embeddings_size=embeddings_size,
        )
        self._2nd_ln = nn.LayerNorm(embeddings_ndim)
        self._ffn = FeedForwardNetwork(embeddings_ndim)

    def forward(self, X):
        X = X + self._mha(self._1st_ln(X))
        X = X + self._ffn(self._2nd_ln(X))
        return X


class NanoNet(nn.Module):
    def __init__(self, vocab_size, embeddings_ndim, self_attention_ndim, context_window_length):
        super().__init__()
        self._tok_embd = nn.Embedding(vocab_size, embeddings_ndim)
        self._pos_embd = nn.Embedding(context_window_length, embeddings_ndim)
        self._atn_blks = nn.Sequential(
            *[SelfAttentionBlock(
                heads_count=hyper_params.heads_count,
                embeddings_ndim=embeddings_ndim,
                embeddings_size=context_window_length,
            ) for _ in range(hyper_params.layer_count)],
        )
        self._lay_norm = nn.LayerNorm(embeddings_ndim)
        self._lin_head = nn.Linear(embeddings_ndim, vocab_size, bias=True)

    def forward(self, X, Y=None):
        _B, T = X.shape
        tok_em = self._tok_embd(X) # B, T, C
        pos = torch.arange(T, device=hyper_params.device)
        pos_em = self._pos_embd(pos) # T, C
        Z = tok_em + pos_em # B, T, C
        Z = self._atn_blks(Z) # B, T, C
        Z = self._lay_norm(Z) # B, T, C
        logits = self._lin_head(Z) # B, T, vocab_size
        if Y is None:
            return logits, None
        _B, _T, C = logits.shape
        logits = logits.view(-1, C)
        target = torch.flatten(Y)
        loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, x, tokens):
        for _ in range(tokens):
            x_cond = (
                x
                if x.size(1) <= hyper_params.context_window_length
                else x[:, -hyper_params.context_window_length :]
            )
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            y_hat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, y_hat), dim=1)
        return x


def do_train(model, data, lrate, max_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
    for step_num in range(max_steps):
        if step_num % hyper_params.eval_interval == 0:
            losses = estimate_loss(model)
            step = step_num + 1
            training_loss = losses["training_data"]
            validation_loss = losses["validation_data"]
            print(f"{step=:4d} {training_loss=:.4f}, {validation_loss=:.4f}")
        Xs_batch, Ys_batch = make_batch(data)
        logits, loss = model(Xs_batch, Ys_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



def main():
    model = NanoNet(
        vocab_size=hyper_params.vocab_size,
        embeddings_ndim=hyper_params.embeddings_ndim,
        self_attention_ndim=hyper_params.embeddings_ndim,
        context_window_length=hyper_params.context_window_length,
    )
    model = model.to(hyper_params.device)
    model.train()
    do_train(
        model,
        data=data_sets.training_data,
        lrate=hyper_params.learning_rate,
        max_steps=hyper_params.max_steps,
    )
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=hyper_params.device)
    print(data_tool.decode(model.generate(context, tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
