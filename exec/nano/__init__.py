import configparser as cp

import torch

from types import SimpleNamespace


torch.manual_seed(1337)


def _get_device():
    for device in [torch.cuda, torch.mps]:
        if device.is_available():
            return device.__name__.split(".")[-1]
    return None

device = _get_device()


_config = cp.ConfigParser()
for config_path in ['configuration.ini',
                    '../configuration.ini',
                    '../../configuration.ini']:
    if len(_config.read(config_path)):
        break


_text = ""
for data_path in ['data/SHAKESPEARE',
                  '../data/SHAKESPEARE',
                  '../../data/SHAKESPEARE']:
    try:
        with open(data_path, 'r', encoding='utf-8') as data_file:
            _text = data_file.read()
            if len(_text):
                break
    except FileNotFoundError:
        continue


_vocab = sorted(list(set(_text)))
_c2i = { c:i for i, c in enumerate(_vocab) }
_i2c = { i:c for i, c in enumerate(_vocab) }
_encode = lambda txt: [_c2i[c] for c in txt]
_decode = lambda vec: ''.join([_i2c[i] for i in vec])

_data = torch.tensor(_encode(_text), dtype=torch.long)
_data_split = int(_config['nano.gpt.data_params'].getfloat('split_ratio') * _data.shape[0])

_batch_size = _config['nano.gpt.data_params'].getint('batch_size')
_context_window_length = _config['nano.gpt.data_params'].getint('context_window_length')

def make_batch(data):
    last_possible_index = (data.shape[0] - _context_window_length) - 1
    random_indexes = torch.randint(last_possible_index + 1, (_batch_size,))
    Xs_batch = torch.stack([data[index:index + _context_window_length]
                            for index in random_indexes])
    Ys_batch = torch.stack([data[index + 1:index + _context_window_length + 1]
                            for index in random_indexes])
    return Xs_batch.to(device), Ys_batch.to(device)

@torch.no_grad()
def estimate_loss(model, data, eval_iters):
    losses = torch.zeros(eval_iters)
    model.eval()
    for eval_iter in range(eval_iters):
        Xs, Ys = make_batch(data)
        logits, loss = model(Xs, Ys)
        losses[eval_iter] = loss.item()
    model.train()
    return losses.mean()


data_sets = SimpleNamespace(
    training_data = _data[:_data_split],
    validation_data = _data[_data_split:]
)

data_tool = SimpleNamespace(
    encode = _encode,
    decode = _decode,
)

data_params = SimpleNamespace(
    vocab_size = len(_vocab),
    batch_size = _batch_size,
    context_window_length = _context_window_length,
)

model_config = SimpleNamespace(
    dropout_ratio = _config['nano.gpt.model_config'].getfloat('dropout_ratio'),
    embeddings_ndim = _config['nano.gpt.model_config'].getint('embeddings_ndim'),
    self_attention_heads_count = _config['nano.gpt.model_config'].getint('self_attention_heads_count'),
    self_attention_block_count = _config['nano.gpt.model_config'].getint('self_attention_block_count'),
)

model_monitor = SimpleNamespace(
    eval_iters = _config['nano.gpt.model_monitor'].getint('eval_iters'),
    eval_interval = _config['nano.gpt.model_monitor'].getint('eval_interval'),
)

hyper_params = SimpleNamespace(
    max_epochs = _config['nano.gpt.hyper_params'].getint('max_epochs'),
    learning_rate = _config['nano.gpt.hyper_params'].getfloat('learning_rate'),
)
