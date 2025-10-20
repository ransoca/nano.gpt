import torch
import torch.nn as nn

from torch.nn import functional as F

from . import hyper_params, data_sets, data_tool
from . import make_batch, estimate_loss


class SingleSelfAttention(nn.Module):
    def __init__(self, ndim, embeddings_ndim, embeddings_size):
        super().__init__()

        self.ndim = ndim

        self.Wk = nn.Linear(embeddings_ndim, ndim, bias=False)
        self.Wq = nn.Linear(embeddings_ndim, ndim, bias=False)
        self.Wv = nn.Linear(embeddings_ndim, ndim, bias=False)

        mask = torch.tril(torch.ones(embeddings_size, embeddings_size))
        self.register_buffer('self_attention_mask', mask)


    def forward(self, Z):
        B, T, C = Z.shape
        K = self.Wk(Z) # B, T, ndim
        Q = self.Wq(Z) # B, T, ndim
        attention_scores = Q @ K.transpose(-2, -1)
        attention_weights = attention_scores / (self.ndim ** -0.5)
        self_attention_weights = attention_weights.masked_fill(
            self.self_attention_mask[:T, :T] == 0,
            float('-inf')
        )
        self_attention_probs = F.softmax(self_attention_weights, dim=1)
        V = self.Wv(Z)
        return self_attention_probs @ V

class NanoNet(nn.Module):
    def __init__(self, vocab_size, embeddings_ndim, self_attention_ndim, context_window_length):
        super().__init__()
        self._tok_embd = nn.Embedding(vocab_size, embeddings_ndim)
        self._pos_embd = nn.Embedding(context_window_length, embeddings_ndim)
        self._atn_head = SingleSelfAttention(
            ndim=self_attention_ndim,
            embeddings_ndim=embeddings_ndim,
            embeddings_size=context_window_length
        )
        self._lin_head = nn.Linear(embeddings_ndim, vocab_size, bias=True)

    def forward(self, X, Y=None):
        _B, T = X.shape
        tok_em = self._tok_embd(X) # B, T, C
        pos = torch.arange(T, device=hyper_params.device)
        pos_em = self._pos_embd(pos) # T, C
        Z = tok_em + pos_em # B, T, C
        Z = self._atn_head(Z) # B, T, C
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
                if x.size(1) <= hyper_params.block_size
                else x[:, -hyper_params.block_size :]
            )
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            y_hat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, y_hat), dim=1)
        return x


def do_train(model, data, lr, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch % hyper_params.eval_interval == 0:
            losses = estimate_loss(model)
            step = epoch + 1
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
        embeddings_ndim=hyper_params.embed_size,
        self_attention_ndim= hyper_params.embed_size,
        context_window_length=hyper_params.block_size,
    )
    model = model.to(hyper_params.device)
    model.train()
    do_train(
        model,
        data=data_sets.training_data,
        lr=hyper_params.learning_rate,
        epochs=hyper_params.training_epochs,
    )
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=hyper_params.device)
    print(data_tool.decode(model.generate(context, tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
