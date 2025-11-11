import spacy
import torch
import random
import torch.nn as nn
import torch.optim as optim
import time

# 新接口所需的附加导入
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from types import SimpleNamespace

SEED = 1234


# TODO 设置全局随机种子，保证结果可复现实验
torch.manual_seed(SEED)
# TODO 让 CuDNN 选择确定性算法，进一步提升可复现性（可能略降速）
torch.backends.cudnn.deterministic = True

# TODO 旧版 Field/LabelField 已移除；新版需使用 tokenizer + vocab + 标签映射
# 使用 spaCy 分词（需提前安装并下载 en_core_web_sm）
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# TODO 使用新版 torchtext 加载原始 IMDB 数据（返回 (label, text) 元组）
train_data = list(IMDB(split='train'))
test_data = list(IMDB(split='test'))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

# TODO 划分验证集；固定随机种子以保证每次划分一致
random.seed(SEED)
random.shuffle(train_data)
valid_ratio = 0.2
n_valid = int(len(train_data) * valid_ratio)
valid_data = train_data[:n_valid]
train_data = train_data[n_valid:]

# TODO 词表大小上限，避免词表过大导致显存/训练效率问题
MAX_VOCAB_SIZE = 25_000

# TODO 从训练集构建词表：仅统计训练集，设置 <unk>/<pad> 特殊符号，并限制最大词表
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    specials=['<unk>', '<pad>'],
    max_tokens=MAX_VOCAB_SIZE
)
vocab.set_default_index(vocab['<unk>'])
PAD_IDX = vocab['<pad>']

# 与旧版打印信息保持一致的近似输出（新版没有 Field.vocab.freqs/itos）
print(f"Unique tokens in TEXT vocabulary: {len(vocab)}")
# 统计训练集的前 20 高频词（用于替代旧版 TEXT.vocab.freqs.most_common(20)）
def token_counter(data_iter):
    c = Counter()
    for label, text in data_iter:
        c.update(tokenizer(text))
    return c
freqs_train = token_counter(train_data)
print(freqs_train.most_common(20))
# 词表前 10 个 token（用于替代旧版 TEXT.vocab.itos[:10]）
try:
    itos = vocab.get_itos()
    print(itos[:10])
except Exception:
    # 兼容没有 get_itos 的情况
    print([vocab.lookup_token(i) for i in range(min(10, len(vocab)))])

# TODO 标签映射到浮点数（BCEWithLogitsLoss 需要 float 标签）
label_to_float = {"neg": 0.0, "pos": 1.0}
# 打印类似旧版 LABEL.vocab.stoi 的信息（用 int 索引展示）
label_stoi = {"neg": 0, "pos": 1}
print(label_stoi)

# 兼容旧代码里对 TEXT/LABEL.vocab 的引用（仅用于打印和获取长度）
TEXT = SimpleNamespace(vocab=vocab)
LABEL = SimpleNamespace(vocab=SimpleNamespace(stoi=label_stoi))

# TODO 训练批大小
BATCH_SIZE = 64

# TODO 选择运行设备（有 GPU 则用 CUDA）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO 新版不再有 BucketIterator；改用 DataLoader + 自定义 collate：
# 1) 将可变长序列转换为索引序列
# 2) 使用 pad_sequence 补齐到等长
# 3) 按 RNN 期望返回形状 [seq_len, batch]，并直接移动到 device
def text_pipeline(text: str):
    return vocab(tokenizer(text))

def label_pipeline(label: str):
    return label_to_float[label]

def collate_batch(batch):
    # batch: List[(label_str, text_str)]
    text_list = []
    label_list = []
    for (label, text) in batch:
        ids = torch.tensor(text_pipeline(text), dtype=torch.long)
        text_list.append(ids)
        label_list.append(label_pipeline(label))
    # pad_sequence 默认返回 [max_len, batch]（batch_first=False）
    padded_text = pad_sequence(text_list, padding_value=PAD_IDX)  # [seq_len, batch]
    labels = torch.tensor(label_list, dtype=torch.float)          # [batch]
    # 直接移动到目标设备，这样训练循环里无需再 .to(device)
    padded_text = padded_text.to(device)
    labels = labels.to(device)
    # 为了最大限度复用你原先的 batch.text / batch.label 写法，这里返回一个具名对象
    return SimpleNamespace(text=padded_text, label=labels)

# 与旧接口的三个 iterator 名字保持一致，便于后续训练代码复用
train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_iterator  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# TODO 定义 RNN 模型（保持与你原来一致的结构）
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    # TODO 前向计算：输入形状 [seq_len, batch]，按最后时间步的隐状态做分类
    def forward(self, text):
        embedded = self.embedding(text)          # [seq_len, batch, emb_dim]
        output, hidden = self.rnn(embedded)      # output: [seq_len, batch, hid_dim]
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))        # [batch, output_dim]


# TODO 词表大小作为模型输入维度
INPUT_DIM = len(TEXT.vocab)
# TODO 词向量维度（可调）
EMBEDDING_DIM = 100
# TODO RNN 隐层维度（可调）
HIDDEN_DIM = 256
# TODO 二分类输出维度
OUTPUT_DIM = 1

# TODO 实例化模型
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# TODO 优化器（学习率可根据收敛速度和稳定性调整）
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# TODO 损失函数：BCEWithLogitsLoss 结合了 Sigmoid + BCE，更稳定
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# TODO 训练一个 epoch：前向 -> 反向 -> 更新；累计损失与准确率
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    # TODO 遍历批次；由于 collate 已经将张量放到 device，这里直接使用 batch.text / batch.label
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# TODO 验证/测试：只前向、累计指标；不计算梯度
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    # TODO 评估阶段关闭梯度，节省显存与加速；按批累计指标
    with torch.no_grad():
        # TODO 遍历验证/测试数据
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# TODO 统计一次 epoch 的时间消耗（分钟、秒）
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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