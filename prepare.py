'''
Preparing the dataset for the machine translation.
'''

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
import urllib
import tiktoken
import tarfile
from collections import Counter
import torchtext.transforms as T
from torchtext.vocab import vocab

# The maximum length for each language. These should be smaller than the block size.
max_len_de = 100
max_len_en = 100

# Download the original dataset.
input_file_path = os.path.join(os.getcwd(), 'dataset/de-en')
if not os.path.exists(input_file_path):
    os.mkdir(input_file_path)
download_tgz = os.path.join(input_file_path, 'train.tgz')
data_url = 'http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz'
urllib.request.urlretrieve(data_url, filename=download_tgz)
with tarfile.open(download_tgz, 'r:gz') as f:
    f.extractall(input_file_path)

input_file_path = os.path.join(os.getcwd(), 'dataset/de-en')
download_tgz = os.path.join(input_file_path, 'val.tgz')
data_url = 'http://data.statmt.org/wmt17/translation-task/dev.tgz'
urllib.request.urlretrieve(data_url, filename=download_tgz)
with tarfile.open(download_tgz, 'r:gz') as f:
    f.extractall(input_file_path)

# Create German dataset.
enc = tiktoken.get_encoding('gpt2')
counter = Counter()
de_train_line_length, de_val_line_length = [], []
de_train_parsed_lines, de_val_parsed_lines = [], []
de_train_tgz = os.path.join(input_file_path, 'training/news-commentary-v12.de-en.de')
de_val_tgz = os.path.join(input_file_path, 'dev/newstest2013.de')
with open(de_train_tgz, newline='\n') as r_de:
    for _, line in enumerate(r_de):
        line = line.replace('\r', ' ').strip()
        parsed_line = [str(b) for b in enc.decode_tokens_bytes(enc.encode_ordinary(line))]
        de_train_parsed_lines.append(parsed_line)
        de_train_line_length.append(len(parsed_line))
        counter.update(parsed_line)
with open(de_val_tgz, newline='\n') as r_de:
    for _, line in enumerate(r_de):
        line = line.replace('\r', ' ').strip()
        parsed_line = [str(b) for b in enc.decode_tokens_bytes(enc.encode_ordinary(line))]
        de_val_parsed_lines.append(parsed_line)
        de_val_line_length.append(len(parsed_line))
        counter.update(parsed_line)

vocab_de = vocab(counter, specials=(['<unk>', '<pad>', '<bos>', '<eos>']))
de_tokenizer = T.Sequential(
    T.VocabTransform(vocab_de),
    T.Truncate(max_len_de - 2), # Subtract for the bos, eos tokens.
    T.AddToken(token=vocab_de['<bos>'], begin=True),
    T.AddToken(token=vocab_de['<eos>'], begin=False),
    T.ToTensor(padding_value=vocab_de['<pad>'])
)

train_de_ids = de_tokenizer(de_train_parsed_lines).numpy()
train_de_ids.tofile(os.path.join(os.getcwd(), 'dataset/de-en_train_de.bin'))

val_de_ids = de_tokenizer(de_val_parsed_lines).numpy()
val_de_ids.tofile(os.path.join(os.getcwd(), 'dataset/de-en_val_de.bin'))

print(f'Size of the German dataset:')
print(f'  vocabulary size: {len(counter):,}')
print(f'  train_de_ids shape: {train_de_ids.shape}')
print(f'  val_de_ids shape: {val_de_ids.shape}')

# Create English dataset. As English data is used as the target, create the shifted version for this.
counter = Counter()
en_train_line_length, en_val_line_length = [], []
en_train_parsed_lines, en_val_parsed_lines = [], []
en_train_tgz = os.path.join(input_file_path, 'training/news-commentary-v12.de-en.en')
en_val_tgz = os.path.join(input_file_path, 'dev/newstest2013.en')
with open(en_train_tgz, newline='\n') as r_en:
    for _, line in enumerate(r_en):
        line = line.replace('\r', ' ').strip()
        parsed_line = [str(b) for b in enc.decode_tokens_bytes(enc.encode_ordinary(line))]
        en_train_parsed_lines.append(parsed_line)
        en_train_line_length.append(len(parsed_line))
        counter.update(parsed_line)
with open(en_val_tgz, newline='\n') as r_en:
    for _, line in enumerate(r_en):
        line = line.replace('\r', ' ').strip()
        parsed_line = [str(b) for b in enc.decode_tokens_bytes(enc.encode_ordinary(line))]
        en_val_parsed_lines.append(parsed_line)
        en_val_line_length.append(len(parsed_line))
        counter.update(parsed_line)

vocab_en = vocab(counter, specials=(['<unk>', '<pad>', '<bos>', '<eos>']))
en_tokenizer = T.Sequential(
    T.VocabTransform(vocab_en),
    T.Truncate(max_len_en - 2), # Subtract for the bos, eos tokens.
    T.AddToken(token=vocab_en['<bos>'], begin=True),
    T.AddToken(token=vocab_en['<eos>'], begin=False),
    T.ToTensor(padding_value=vocab_en['<pad>'])
)
en_tokenizer_shifted = T.Sequential(
    T.VocabTransform(vocab_en),
    T.Truncate(max_len_en - 2), # Subtract for the eos token.
    T.AddToken(token=vocab_en['<eos>'], begin=False),
    T.ToTensor(padding_value=vocab_en['<pad>'])
)

train_en_ids = en_tokenizer(en_train_parsed_lines).numpy()
train_en_ids.tofile(os.path.join(os.getcwd(), 'dataset/de-en_train_en.bin'))
train_en_ids_shifted = en_tokenizer_shifted(en_train_parsed_lines).numpy()
train_en_ids_shifted.tofile(os.path.join(os.getcwd(), 'dataset/de-en_train_sft_en.bin'))

val_en_ids = en_tokenizer(en_val_parsed_lines).numpy()
val_en_ids.tofile(os.path.join(os.getcwd(), 'dataset/de-en_val_en.bin'))
val_en_ids_shifted = en_tokenizer_shifted(en_val_parsed_lines).numpy()
val_en_ids_shifted.tofile(os.path.join(os.getcwd(), 'dataset/de-en_val_sft_en.bin'))

print(f'Size of the English dataset:')
print(f'  vocabulary size: {len(counter):,}')
print(f'  train_en_ids shape: {train_en_ids.shape}')
print(f'  val_en_ids shape: {val_en_ids.shape}')

# Remove the unnecessary files.
shutil.rmtree(input_file_path)