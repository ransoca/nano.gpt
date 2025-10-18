import torch

from types import SimpleNamespace

torch.manual_seed(1337)

with open('./data/SHAKESPEARE', 'r', encoding='utf-8') as data_file:
  _text = data_file.read()

_vocab = sorted(list(set(_text)))
_c2i = { c:i for i, c in enumerate(_vocab) }
_i2c = { i:c for i, c in enumerate(_vocab) }

data_tool = SimpleNamespace(
    encode = lambda txt: [_c2i[c] for c in txt],
    decode = lambda vec: ''.join([_i2c[i] for i in vec])
)

def _get_device():
    for device in [torch.cuda, torch.mps]:
        if device.is_available():
            return device.__name__.split(".")[-1]
    return None

hyper_params = SimpleNamespace(
    device = _get_device(),
    batch_size = 32,
    block_size = 8,
    eval_iters = 200,
    eval_interval = 300,
    learning_rate = 1e-3,
    training_epochs = 10**4,
    vocab_size = len(_vocab),
)

_data = torch.tensor(data_tool.encode(_text), dtype=torch.long)
_data_split = int(0.9 * _data.shape[0])

data_sets = SimpleNamespace(
    training_data = _data[:_data_split],
    validation_data = _data[_data_split:]
)

def make_batch(data):
    last_possible_index = (data.shape[0] - hyper_params.block_size) - 1
    random_indexes = torch.randint(last_possible_index + 1, (hyper_params.batch_size,))
    Xs_batch = torch.stack([data[index:index + hyper_params.block_size] for index in random_indexes])
    Ys_batch = torch.stack([data[index + 1:index + hyper_params.block_size + 1] for index in random_indexes])
    return Xs_batch.to(hyper_params.device), Ys_batch.to(hyper_params.device)

@torch.no_grad()
def estimate_loss(model):
    output = dict()
    model.eval()
    for data_name, data in vars(data_sets).items():
        losses = torch.zeros(hyper_params.eval_iters)
        for eval_iter in range(hyper_params.eval_iters):
            Xs_batch, Ys_batch = make_batch(data)
            logits, loss = model(Xs_batch, Ys_batch)
            losses[eval_iter] = loss.item()
        output[data_name] = losses.mean()
    model.train()
    return output
