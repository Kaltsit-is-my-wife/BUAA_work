# 必须最先静音 torchtext 弃用警告（在导入任何 torchtext 子模块之前）
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# 可选：在运行前检查 IMDB 下载缓存所需的 portalocker
try:
    import portalocker  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "Missing dependency: portalocker (IMDB 数据下载/缓存需要文件锁)。\n"
        "请先安装：\n"
        "  - conda install -c conda-forge portalocker\n"
        "或\n"
        "  - python -m pip install 'portalocker>=2.0.0'\n"
        f"原始错误：{e}"
    )

import torch
import random
import time
from types import SimpleNamespace
from collections import Counter

import torch.nn as nn
import torch.optim as optim

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


SEED = 1234

# TODO 设置随机种子，保证可复现实验
torch.manual_seed(SEED)
# TODO 让 CuDNN 选择确定性算法（可能略降速）
torch.backends.cudnn.deterministic = True

# 设备与加速相关开关
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()  # GPU 下启用 AMP
USE_BENCHMARK = torch.cuda.is_available()  # 可选：GPU 下开启算法搜索
if USE_BENCHMARK:
    # TODO 提升卷积/RNN选择最优算法性能（牺牲严格可复现性）
    torch.backends.cudnn.benchmark = True

print('device:', device)

# TODO 旧版 Field/LabelField 已移除；新版需使用 tokenizer + vocab + 标签映射
# 优先用 spaCy 的英文模型；若缺失则回退 basic_english，保证代码可运行
try:
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
except Exception as e:
    print("spaCy model not found or failed to load, fallback to basic_english:", e)
    tokenizer = get_tokenizer('basic_english')
print('tokenizer:', tokenizer)


def yield_tokens(data_iterable):
    for label, text in data_iterable:
        yield tokenizer(text)


# 加载数据；注意：IMDB(split='train') 返回一次性可迭代对象
train_data_all = list(IMDB(split='train'))
test_data = list(IMDB(split='test'))

# TODO 词表大小上限，避免显存/效率问题
MAX_VOCAB_SIZE = 25_000

# TODO 从训练集构建词表：仅统计训练集，设置 <unk>/<pad> 特殊符号，并限制最大词表
vocab = build_vocab_from_iterator(
    yield_tokens(train_data_all),
    specials=['<unk>', '<pad>'],
    max_tokens=MAX_VOCAB_SIZE
)
vocab.set_default_index(vocab['<unk>'])
PAD_IDX = vocab['<pad>']

print(f"Unique tokens in TEXT vocabulary: {len(vocab)}")

# 统计训练集的前 20 高频词（用于替代旧版 TEXT.vocab.freqs.most_common(20)）
def token_counter(data_iterable):
    c = Counter()
    for label, text in data_iterable:
        c.update(tokenizer(text))
    return c

freqs_train = token_counter(train_data_all)
print(freqs_train.most_common(20))

# 词表前 10 个 token（用于替代旧版 TEXT.vocab.itos[:10]）
try:
    itos = vocab.get_itos()
    print(itos[:10])
except Exception:
    print([vocab.lookup_token(i) for i in range(min(10, len(vocab)))])

# TODO 标签映射到浮点数（BCEWithLogitsLoss 需要 float 标签）
# 兼容 'neg'/'pos'、0/1 以及 1/2 三种常见格式
def label_pipeline(x):
    # 若是 bytes，尝试解码
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8")
        except Exception:
            pass

    # 字符串标签
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ("neg", "negative"):
            return 0.0
        if xl in ("pos", "positive"):
            return 1.0
        if xl.isdigit():
            xi = int(xl)
            if xi in (0, 1):
                return float(xi)
            if xi in (1, 2):
                return float(xi - 1)  # 1->0.0, 2->1.0
        raise ValueError(f"Unrecognized string label: {x!r}")

    # 数值标签
    if isinstance(x, (int, float)):
        xi = int(x)
        if xi in (0, 1):
            return float(xi)
        if xi in (1, 2):
            return float(xi - 1)      # 1->0.0, 2->1.0
        raise ValueError(f"Unrecognized numeric label: {x!r}")

    # 其它类型（极少见）
    raise TypeError(f"Unsupported label type: {type(x)} value={x!r}")

# 打印类似旧版 LABEL.vocab.stoi 的信息（用于提示）
label_stoi = {"neg": 0, "pos": 1}
print(label_stoi)

# TODO 划分验证集；固定随机种子以保证每次划分一致
random.seed(SEED)
random.shuffle(train_data_all)
valid_ratio = 0.2
n_valid = int(len(train_data_all) * valid_ratio)
valid_data = train_data_all[:n_valid]
train_data = train_data_all[n_valid:]

# TODO 训练批大小（GPU 可适当调大，如 128）
BATCH_SIZE = 128 if torch.cuda.is_available() else 64

# 文本处理管线
def text_pipeline(text: str):
    return vocab(tokenizer(text))

# TODO 新版不再有 BucketIterator；改用 DataLoader + 自定义 collate：
# 1) 将可变长序列转换为索引序列
# 2) 使用 pad_sequence 补齐到等长
# 注意：为让 pin_memory 生效，这里保持张量在 CPU，训练时再搬到 GPU
def collate_batch(batch):
    text_list = []
    label_list = []
    for (label, text) in batch:
        ids = torch.tensor(text_pipeline(text), dtype=torch.long)
        if ids.numel() == 0:
            ids = torch.tensor([vocab['<unk>']], dtype=torch.long)
        text_list.append(ids)
        label_list.append(label_pipeline(label))
    # 这里用 batch_first=False，保持与原 RNN 输入一致 [seq_len, batch]
    padded_text = pad_sequence(text_list, padding_value=PAD_IDX)  # CPU [seq_len, batch]
    labels = torch.tensor(label_list, dtype=torch.float)          # CPU [batch]
    return SimpleNamespace(text=padded_text, label=labels)

# DataLoader 参数：GPU 下启用 pin_memory、适度的 num_workers
loader_kwargs = {}
if torch.cuda.is_available():
    loader_kwargs.update(dict(pin_memory=True, num_workers=2, persistent_workers=False))

train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_batch, **loader_kwargs)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_batch, **loader_kwargs)
test_iterator  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_batch, **loader_kwargs)

# TODO 定义 RNN 模型（保持与你原来一致的结构）
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)  # 默认 batch_first=False
        self.fc = nn.Linear(hidden_dim, output_dim)

    # TODO 前向计算：输入形状 [seq_len, batch]，按最后时间步的隐状态做分类
    def forward(self, text):
        embedded = self.embedding(text)          # [seq_len, batch, emb_dim]
        output, hidden = self.rnn(embedded)      # output: [seq_len, batch, hid_dim]
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))        # [batch, output_dim]

# TODO 词表大小作为模型输入维度
INPUT_DIM = len(vocab)
# TODO 词向量维度（可调）
EMBEDDING_DIM = 100
# TODO RNN 隐层维度（可调）
HIDDEN_DIM = 256
# TODO 二分类输出维度
OUTPUT_DIM = 1

# TODO 实例化模型
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

# 可选：在 PyTorch 2.x 下尝试编译加速（对 RNN 提升有限；失败会安全降级）
try:
    # model = torch.compile(model)  # 需要 PyTorch 2.x
    print("torch.compile enabled")
except Exception as _:
    print("torch.compile not available or failed; continue without it")

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# TODO 优化器（Adam 通常更快收敛；你也可换回 SGD）
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TODO 损失函数：BCEWithLogitsLoss 结合了 Sigmoid + BCE，更稳定
criterion = nn.BCEWithLogitsLoss().to(device)

# AMP 组件
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# TODO 训练一个 epoch：前向 -> 反向 -> 更新；使用 AMP + non_blocking 拷贝
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()
    for batch in iterator:
        # 将 batch 数据搬到 GPU（若可用）
        text = batch.text.to(device, non_blocking=True)
        labels = batch.label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# TODO 验证/测试：只前向、累计指标；在 GPU 上也用 autocast 加速
def evaluate(model, iterator, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
        for batch in iterator:
            text = batch.text.to(device, non_blocking=True)
            labels = batch.label.to(device, non_blocking=True)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# TODO 统计一次 epoch 的时间消耗（分钟、秒）
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 单句预测
def predict_sentiment(model, sentence: str):
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.float16):
        ids = torch.tensor(vocab(tokenizer(sentence)), dtype=torch.long)
        if ids.numel() == 0:
            ids = torch.tensor([vocab['<unk>']], dtype=torch.long)
        ids = ids.unsqueeze(1).to(device, non_blocking=True)  # [seq_len] -> [seq_len, 1]
        prob = torch.sigmoid(model(ids)).item()
    return prob  # >=0.5 视为 positive

if __name__ == '__main__':
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    print("training start")

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # TODO 保存在验证集上表现最好的模型参数（可在测试前加载）
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # 加载最佳权重并在测试集上评估
    model.load_state_dict(torch.load('tut2-model.pt', map_location=device))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'[Best Model] Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # 示例预测
    demo_sentences = [
        "This movie was absolutely wonderful, I loved it.",
        "Terrible boring film. I will never watch it again.",
        "the film is too terrible!",
        "the film is so interesting!",
        "i love the film!",
        "the film is 666!",
        "god damn it, it is so cool!"
    ]
    for s in demo_sentences:
        score = predict_sentiment(model, s)
        print(f'"{s}" => {score:.4f} ({"Positive" if score >= 0.5 else "Negative"})')
