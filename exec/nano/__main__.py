import torch
import torch.nn as nn

from torch.nn import functional as F

from . import device
from . import data_sets, data_tool, data_params
from . import model_config, model_monitor
from . import hyper_params
from . import make_batch, estimate_loss

from pprint import pprint


class SingleSelfAttention(nn.Module):
    def __init__(self, ndim):
        super().__init__()

        self._ndim = ndim

        self._Wk = nn.Linear(model_config.embeddings_ndim, ndim, bias=False)
        self._Wq = nn.Linear(model_config.embeddings_ndim, ndim, bias=False)
        self._Wv = nn.Linear(model_config.embeddings_ndim, ndim, bias=False)

        self._dropout = nn.Dropout(model_config.dropout_ratio)

        tril = torch.tril(torch.ones(data_params.context_window_length,
                                     data_params.context_window_length))
        self.register_buffer('_mask', tril)


    def forward(self, X):
        _, T, _ = X.shape # B, T, C
        K = self._Wk(X) # B, T, ndim
        Q = self._Wq(X) # B, T, ndim
        attention_scores = Q @ K.transpose(-2, -1)
        attention_weights = attention_scores * self._ndim ** -0.5
        self_attention_weights = attention_weights.masked_fill(
            self._mask[:T, :T] == 0,
            float('-inf')
        )
        self_attention_probs = F.softmax(self_attention_weights, dim=1)
        self_attention_probs = self._dropout(self_attention_probs)
        V = self._Wv(X)
        X = self_attention_probs @ V
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, head_ndim):
        super().__init__()
        self._heads = nn.ModuleList(
            [SingleSelfAttention(head_ndim)
             for _ in range(model_config.self_attention_heads_count)]
        )
        self._projs = nn.Linear(head_ndim * model_config.self_attention_heads_count,
                                model_config.embeddings_ndim)
        self._dropout = nn.Dropout(model_config.dropout_ratio)

    def forward(self, X):
        X = torch.cat([head(X) for head in self._heads], dim=-1)
        X = self._projs(X)
        X = self._dropout(X)
        return X


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 4 * model_config.embeddings_ndim
        self._net = nn.Sequential(
            nn.Linear(model_config.embeddings_ndim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, model_config.embeddings_ndim),
            nn.Dropout(model_config.dropout_ratio)
        )

    def forward(self, X):
        X = self._net(X)
        return X


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_ndim = model_config.embeddings_ndim // model_config.self_attention_heads_count
        self._1st_ln = nn.LayerNorm(model_config.embeddings_ndim)
        self._mha = MultiHeadAttention(head_ndim)
        self._2nd_ln = nn.LayerNorm(model_config.embeddings_ndim)
        self._ffn = FeedForwardNetwork()

    def forward(self, X):
        X = X + self._mha(self._1st_ln(X))
        X = X + self._ffn(self._2nd_ln(X))
        return X


class NanoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._tok_embd = nn.Embedding(data_params.vocab_size, model_config.embeddings_ndim)
        self._pos_embd = nn.Embedding(data_params.context_window_length, model_config.embeddings_ndim)
        self._atn_blks = nn.Sequential(*[SelfAttentionBlock()
                                         for _ in range(model_config.self_attention_block_count)])
        self._lay_norm = nn.LayerNorm(model_config.embeddings_ndim)
        self._lin_head = nn.Linear(model_config.embeddings_ndim, data_params.vocab_size)

    def forward(self, X, Y=None):
        _, T = X.shape
        tok_emb = self._tok_embd(X) # B, T, C
        pos_emb = self._pos_embd(torch.arange(T, device=device)) # T, C
        X = tok_emb + pos_emb # B, T, C
        X = self._atn_blks(X) # B, T, C
        X = self._lay_norm(X) # B, T, C
        logits = self._lin_head(X) # B, T, vocab_size
        if Y is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = Y.view(B * T)
        loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, x, tokens):
        for _ in range(tokens):
            x_cond = x[:, -data_params.context_window_length:]
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x


def do_train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)
    for epoch in range(hyper_params.max_epochs):
        if epoch % model_monitor.eval_interval == 0:
            training_loss = estimate_loss(model, data_sets.training_data, model_monitor.eval_iters)
            validation_loss = estimate_loss(model, data_sets.validation_data, model_monitor.eval_iters)
            print(f"{epoch=:4d} {training_loss=:.4f}, {validation_loss=:.4f}")
        Xs, Ys = make_batch(data_sets.training_data)
        logits, loss = model(Xs, Ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    model = NanoNet()
    model = model.to(device)
    print(model)
    print(sum(param.numel() for param in model.parameters())/1e6, 'M parameters')

    model.train()
    do_train(model)

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(data_tool.decode(model.generate(context, tokens=500)[0].tolist()))


if __name__ == "__main__":
    pprint(data_params)
    pprint(hyper_params)
    pprint(model_config)
    main()
