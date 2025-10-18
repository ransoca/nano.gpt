import torch
import torch.nn as nn

from torch.nn import functional as F

from . import hyper_params, data_sets, data_tool
from . import make_batch, estimate_loss


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self._embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, X, Y=None):
        logits = self._embeddings(X)
        if Y is None:
            return logits, None
        _B, _T, C = logits.shape
        logits = logits.view(-1, C)
        target = torch.flatten(Y)
        loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, x, tokens):
        for _ in range(tokens):
            logits, loss = self(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            y_hat = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, y_hat), dim=1)
        return x


def do_train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params.learning_rate)
    for epoch in range(hyper_params.training_epochs):
        if epoch % hyper_params.eval_interval == 0:
            losses = estimate_loss(model)
            step = epoch + 1
            training_loss = losses["training_data"]
            validation_loss = losses["validation_data"]
            print(f"{step=:4d} {training_loss=:.4f}, {validation_loss=:.4f}")
        Xs_batch, Ys_batch = make_batch(data_sets.training_data)
        logits, loss = model(Xs_batch, Ys_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    model = BigramLM(hyper_params.vocab_size)
    model = model.to(hyper_params.device)
    do_train(model)
    context = torch.zeros((1, 1), dtype=torch.long, device=hyper_params.device)
    print(data_tool.decode(model.generate(context, tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
