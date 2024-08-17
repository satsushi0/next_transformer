'''
Training script for the encoder-decoder model for the machine translation.
Borrowed mostly from nanoGPT/train.py: https://github.com/karpathy/nanoGPT/blob/master/train.py
'''

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import EncDecModel, ModelConfig

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True
init_from = 'scratch' # Either 'scratch' or 'resume'

wandb_log = False
wandb_project = 'next_transformer'
wandb_run_name = 'de-en MT' # 'run' + str(time.time())

dataset_dir = 'dataset'
gradient_accumulation_steps = 8 * 5
enc_vocab_size = 20200
dec_vocab_size = 40000
batch_size = 12
block_size = 100 # This should be the same as the maximum sequence length of the dataset.
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
activation = "gelu"
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

backend = 'nccl' # DDP settings
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported else 'float16'
compile = True

config_keys = [key for key, val in globals().items() if not key.startswith('_') and isinstance(val, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {key: globals()[key] for key in config_keys}


# -----------------------------------------------------------------------------
# Setups
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print('Tokens per iteration will be: {tokens_per_iter:,}')

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join(os.getcwd(), dataset_dir)
def get_batch(split: str) -> torch.Tensor | torch.Tensor:
    if split == 'train':
        data_de = np.memmap(os.path.join(data_dir, 'de-en_train_de.bin'), dtype=np.uint16, mode='r')
        data_en = np.memmap(os.path.join(data_dir, 'de-en_train_en.bin'), dtype=np.uint16, mode='r')
        data_en_sft = np.memmap(os.path.join(data_dir, 'de-en_train_sft_en.bin'), dtype=np.uint16, mode='r')
    else:
        data_de = np.memmap(os.path.join(data_dir, 'de-en_val_de.bin'), dtype=np.uint16, mode='r')
        data_en = np.memmap(os.path.join(data_dir, 'de-en_val_en.bin'), dtype=np.uint16, mode='r')
        data_en_sft = np.memmap(os.path.join(data_dir, 'de-en_val_sft_en.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data_de), (batch_size,))
    enc_x = torch.from_numpy((data_de[ix, :]).astype(np.int64))
    dec_x = torch.from_numpy((data_en[ix, :]).astype(np.int64))
    dec_y = torch.from_numpy((data_en_sft[ix, :]).astype(np.int64))
    if device_type == 'cuda':
        enc_x = enc_x.pin_memory().to(device, non_blocking=True)
        dec_x = dec_x.pin_memory().to(device, non_blocking=True)
        dec_y = dec_y.pin_memory().to(device, non_blocking=True)
    else:
        enc_x, dec_x, dec_y = enc_x.to(device), dec_x.to(device), dec_y.to(device)
    return enc_x, dec_x, dec_y
    

# -----------------------------------------------------------------------------
# Initializations
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, 
                  enc_vocab_size=enc_vocab_size, dec_vocab_size=dec_vocab_size, dropout=dropout, 
                  activation=activation)

if init_from == 'scratch':
    print(f'Iitializing a new model from scratch.')
    model_config = ModelConfig(**model_args)
    model = EncDecModel(model_config)
else: # init_from == 'resume'
    print(f'Resuming training from {out_dir}.')
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'enc_vocab_size', 'dec_vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model_config = ModelConfig(**model_args)
    model = EncDecModel(model_config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith((unwanted_prefix)):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print(f'Compiling the model... (takes a ~min)')
    unoptimized_model = model
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss() -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluation step.
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f'Step {iter_num:,}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
                'lr': lr,
                'mfu': running_mfu * 100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f'Saving checkpoint to {out_dir}.')

    # Gradient Descent.
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f'Iter {iter_num:,}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%')
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
