"""
Train the Kangaroo adapter for Qwen2.5-VL self-speculative decoding.

Trains a lightweight adapter to predict the full model's output distribution
from early-exit layer hidden states, using KL-divergence loss.

Usage:
    accelerate launch train_adapter.py \
        --basepath Qwen/Qwen2.5-VL-3B-Instruct \
        --datadir ./kangaroo/training_data/ \
        --outdir ./kangaroo/adapter_checkpoints/ \
        --exit_layer 2 \
        --lr 3e-5 --bs 4

Adapted from Kangaroo's train.py for Qwen2.5-VL architecture.
"""

import argparse

parser = argparse.ArgumentParser(description='Train Kangaroo adapter for Qwen2.5-VL')
parser.add_argument('--basepath', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                    help='Path to base model (for config and lm_head weights)')
parser.add_argument('--datadir', type=str, required=True,
                    help='Directory containing training data (.ckpt files)')
parser.add_argument('--outdir', type=str, required=True,
                    help='Output directory for adapter checkpoints')
parser.add_argument('--exit_layer', type=int, default=2,
                    help='Early exit layer index')
parser.add_argument('--num_adapter_layers', type=int, default=1,
                    help='Number of transformer layers in the adapter')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--num_warmup_steps', type=int, default=2000)
parser.add_argument('--total_steps', type=int, default=800000)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--log_steps', type=int, default=20)
args = parser.parse_args()

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoConfig, get_linear_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True

from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=args.gradient_accumulation_steps,
)

from adapter import AdapterModel, create_adapter_config

# Attempt TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(args.outdir, "tensorboard"))
    else:
        writer = None
except ImportError:
    writer = None

# ========== Load LM Head (frozen) ==========
base_config = AutoConfig.from_pretrained(args.basepath)
head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

# Load lm_head weights from base model
try:
    from safetensors import safe_open
    index_path = os.path.join(args.basepath, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except Exception:
    try:
        index_path = os.path.join(args.basepath, "pytorch_model.bin.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path), map_location='cpu')
        tensor = weights["lm_head.weight"].float()
    except Exception:
        # Try loading from single file
        model_path = os.path.join(args.basepath, "model.safetensors")
        if os.path.exists(model_path):
            with safe_open(model_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor("lm_head.weight").float()
        else:
            raise RuntimeError(f"Cannot find lm_head weights in {args.basepath}")

head.weight.data = tensor
head.eval()
for param in head.parameters():
    param.requires_grad = False


# ========== Dataset ==========
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith('.ckpt'):
                datapath.append(os.path.join(root, file))
    return sorted(datapath)


class AdapterDataset(Dataset):
    def __init__(self, datapath, exit_layer, max_len):
        self.data = datapath
        self.exit_layer = exit_layer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location='cpu')
        new_data = {}

        hidden_state = data['hidden_state'][:self.max_len][None, :]
        input_ids = data['input_ids'][:self.max_len][None, :]
        loss_mask = data['loss_mask'][:self.max_len][None, :]

        hidden_state_early = data[f'hidden_state_layer{self.exit_layer}'][:self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        # Shift targets: predict next token
        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)

        hidden_state_early = hidden_state_early[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        hidden_state_early = torch.cat((hidden_state_early, zeropadding), dim=1)

        # Shift loss_mask to align with shifted hidden states:
        # hidden_state_early[i] = original position i+1, so loss_mask should also shift
        loss_mask = loss_mask[1:] + [0]
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["hidden_state_early"] = hidden_state_early

        return new_data


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        return torch.cat((intensors, padding_tensor), dim=1)

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states_early = torch.cat([self.paddingtensor(item['hidden_state_early'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        return {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "hidden_states_early": batch_hidden_states_early,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


# ========== Setup data ==========
datapath = list_files(args.datadir)
if len(datapath) == 0:
    raise ValueError(f"No .ckpt files found in {args.datadir}")

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
print(f"Training: {len(traindatapath)} samples, Testing: {len(testdatapath)} samples")

traindataset = AdapterDataset(traindatapath, args.exit_layer, args.max_len)
testdataset = AdapterDataset(testdatapath, args.exit_layer, args.max_len)

# Diagnostic: check loss_mask coverage in training data
if accelerator.is_main_process:
    print("\n[Data Diagnostic] Checking loss_mask in first 20 training samples...")
    num_empty = 0
    for idx in range(min(20, len(traindataset))):
        sample = traindataset[idx]
        mask_sum = sum(sample["loss_mask"])
        mask_len = len(sample["loss_mask"])
        if mask_sum == 0:
            num_empty += 1
            # Load raw data for debugging
            raw = torch.load(traindatapath[idx], map_location='cpu')
            raw_mask_sum = raw['loss_mask'].sum().item()
            print(f"  Sample {idx}: loss_mask all ZERO (len={mask_len}, "
                  f"raw_mask_sum={raw_mask_sum:.0f})")
        else:
            print(f"  Sample {idx}: loss_mask has {mask_sum:.0f}/{mask_len} active positions")
    if num_empty > 0:
        print(f"  WARNING: {num_empty}/20 samples have empty loss_mask! "
              f"Training will be ineffective.")
        print(f"  → Check that your training data has assistant responses")
        print(f"  → If raw_mask_sum is also 0, regenerate data with fixed build_loss_mask()")
        print(f"  → If raw_mask_sum > 0 but mask is 0 after processing, "
              f"check max_len={args.max_len} truncation")
    print()

train_loader = DataLoader(
    traindataset, batch_size=args.bs, shuffle=True,
    collate_fn=DataCollatorWithPadding(), num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    testdataset, batch_size=args.bs, shuffle=False,
    collate_fn=DataCollatorWithPadding(), num_workers=4, pin_memory=True,
)

# ========== Setup model ==========
if accelerator.is_main_process:
    os.makedirs(args.outdir, exist_ok=True)

adapter_config = create_adapter_config(args.basepath, num_adapter_layers=args.num_adapter_layers)
model = AdapterModel(adapter_config)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps,
)

model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, head, optimizer, train_loader, test_loader, scheduler,
)

if args.start_epoch > 0:
    state_dir = os.path.join(args.outdir, f"state_{args.start_epoch - 1}")
    if os.path.exists(state_dir):
        accelerator.load_state(state_dir)
        print(f"Resumed from {state_dir}")


# ========== Training Loop ==========
for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
    print(f"=== Epoch {epoch} ===")
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        from torch.autograd import Variable
        data["hidden_states_early"] = Variable(data["hidden_states_early"], requires_grad=True)
        predict = model(inputs_embeds=data["hidden_states_early"], attention_mask=data["attention_mask"])

        with torch.no_grad():
            target_head = head(data["target"])
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        out_head = head(predict)
        prob_exit = F.softmax(out_head, dim=2)
        prob_last = F.softmax(target_head, dim=2)
        prob_acc = torch.min(prob_last, prob_exit).sum(dim=2)

        out_logp = nn.LogSoftmax(dim=2)(out_head)
        loss_mask = data["loss_mask"][:, :, None]
        mask_sum = loss_mask.sum()
        if mask_sum == 0:
            # Skip batches with no supervised positions to avoid 0/0 NaN
            continue
        plogp = target_p * out_logp
        loss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / mask_sum
        prob_acc = torch.sum(data["loss_mask"] * prob_acc) / data["loss_mask"].sum()

        if accelerator.is_main_process and batch_idx % args.log_steps == 0:
            print(f"\nStep: {batch_idx}\tLR: {optimizer.optimizer.param_groups[0]['lr']:.6f}"
                  f"\tAccept: {prob_acc.item():.4f}\tLoss: {loss.item():.4f}")

        accelerator.backward(loss)
        accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if loss != loss and accelerator.is_main_process:
            print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
            break

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head_flat = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target_flat = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head_flat, target_flat, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc

        if accelerator.is_main_process and writer is not None and ct != 0:
            global_step = batch_idx + len(train_loader) * epoch
            writer.add_scalar("train/lr", optimizer.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/prob_accept", prob_acc.item(), global_step)
            writer.add_scalar("train/accuracy", cc / ct, global_step)
            for id_i, acc_i in enumerate(topkacc):
                writer.add_scalar(f"train/top_{id_i + 1}_acc", acc_i.item() / ct, global_step)

        epoch_loss += loss.item()
        num_batches += 1

    # Epoch summary
    correct_t, total_t = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
    correct_val, total_val = correct_t.sum().item(), total_t.sum().item()
    epoch_loss /= max(num_batches, 1)

    if accelerator.is_local_main_process:
        print(f'Epoch [{epoch + 1}/{args.start_epoch + args.num_epochs}], Loss: {epoch_loss:.4f}')
        if total_val > 0:
            print(f'Train Accuracy: {100 * correct_val / total_val:.2f}%')

    # Save checkpoint
    if epoch % args.save_freq == 0 or epoch == args.start_epoch + args.num_epochs - 1:
        accelerator.save_state(output_dir=os.path.join(args.outdir,"state", f"state_{epoch}"))

        # Also save adapter weights in a standalone format
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            adapter_save_dir = os.path.join(args.outdir,"epoch",f"adapter_epoch_{epoch}")
            os.makedirs(adapter_save_dir, exist_ok=True)
            torch.save(unwrapped_model.state_dict(), os.path.join(adapter_save_dir, "adapter_model.bin"))
            # Save adapter config
            adapter_config_dict = {
                'hidden_size': adapter_config.hidden_size,
                'num_attention_heads': adapter_config.num_attention_heads,
                'num_key_value_heads': adapter_config.num_key_value_heads,
                'intermediate_size': adapter_config.intermediate_size,
                'num_hidden_layers': adapter_config.num_hidden_layers,
                'rms_norm_eps': adapter_config.rms_norm_eps,
                'vocab_size': adapter_config.vocab_size,
                'max_position_embeddings': adapter_config.max_position_embeddings,
                'exit_layer': args.exit_layer,
            }
            with open(os.path.join(adapter_save_dir, "adapter_config.json"), 'w') as f:
                json.dump(adapter_config_dict, f, indent=2)
            print(f"Saved adapter checkpoint to {adapter_save_dir}")

if accelerator.is_main_process and writer is not None:
    writer.close()

print("Training complete!")
