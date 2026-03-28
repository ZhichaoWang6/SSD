"""
Generate training data for the Kangaroo adapter by collecting hidden states
from the full Qwen2.5-VL model on MMDuet2's multimodal data.

For each sample, saves:
- input_ids: tokenized input sequence
- loss_mask: which positions contribute to the loss (assistant responses only)
- hidden_state_layerN: hidden states from early exit layer N
- hidden_state: hidden states from the final layer (before norm)

Usage:
    python generate_training_data.py \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path ./data/annotations/ego-frame_input_format.json \
        --output_dir ./training_data/ \
        --exit_layers 2,3 \
        --start 0 --end 100
"""

import argparse
import json
import os
import copy

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description='Generate adapter training data')
    parser.add_argument('--model_path', type=str, default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
    parser.add_argument('--data_path', type=str, default="/data/wangzhichao/projects/SSD/train_data_test.json",
                        help='Path to proactive eval data (json) or SFT data (jsonl)')
    parser.add_argument('--output_dir', type=str, default='./kangaroo/training_data/')
    parser.add_argument('--exit_layers', type=str, default='2,3',
                        help='Comma-separated list of exit layers to save hidden states for')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help='Maximum sequence length to process')
    parser.add_argument('--frame_interval', type=float, default=1.0,
                        help='Seconds per frame (default: 1.0 for 1_sec_per_frame)')
    return parser.parse_args()


def load_data(data_path):
    """Load data from json or jsonl format."""
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(data_path) as f:
            data = json.load(f)
    return data


def build_loss_mask(input_ids, tokenizer):
    """
    Create loss mask that only includes assistant response tokens.
    Uses the chat template's assistant token markers to identify response regions.
    """
    loss_mask = torch.zeros_like(input_ids[0], dtype=torch.float32)

    # Find assistant response regions by looking for the pattern after
    # the assistant header token. For Qwen2.5, assistant turns end with <|im_end|>
    im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    ids = input_ids[0].tolist()
    in_assistant = False
    for i, token_id in enumerate(ids):
        if token_id == im_start_id:
            # Check if next token indicates 'assistant'
            # The pattern is: <|im_start|>assistant\n...
            # We'll mark everything after the newline until <|im_end|>
            in_assistant = False  # reset
            # Look ahead for 'assistant' token
            if i + 1 < len(ids):
                # Decode the next few tokens to check role
                snippet = tokenizer.decode(ids[i+1:min(i+10, len(ids))], skip_special_tokens=False)
                if snippet.startswith('assistant'):
                    in_assistant = True
                    continue
        elif token_id == im_end_id:
            if in_assistant:
                in_assistant = False
            continue

        if in_assistant:
            loss_mask[i] = 1.0

    return loss_mask


@torch.no_grad()
def process_single_example(model, processor, example, exit_layers, max_seq_len, frame_interval):
    """
    Process a single example through the full model and collect hidden states.

    Args:
        model: Full Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        example: Data example with 'conversation' field
        exit_layers: List of exit layer indices to save
        max_seq_len: Maximum sequence length
        frame_interval: Seconds per frame

    Returns:
        dict with input_ids, loss_mask, hidden_state_layerN, hidden_state
        or None if the example can't be processed
    """
    conversation = example.get('conversation', example.get('messages', []))
    if not conversation:
        return None

    # Build conversation history
    history = []
    for turn in conversation:
        history.append(turn)

    # Apply chat template
    text = processor.apply_chat_template(
        history, tokenize=False, add_generation_prompt=False,
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(history)

    # Tokenize and process
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Truncate if too long
    if inputs.input_ids.shape[1] > max_seq_len:
        inputs['input_ids'] = inputs['input_ids'][:, :max_seq_len]
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_seq_len]

    # Forward pass with all hidden states
    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'output_hidden_states': True,
        'return_dict': True,
        'use_cache': False,
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    try:
        outputs = model(**forward_kwargs)
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

    # Collect results
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    loss_mask = build_loss_mask(inputs['input_ids'], tokenizer)

    result = {
        'input_ids': inputs['input_ids'].cpu()[0],
        'loss_mask': loss_mask.cpu(),
    }

    # Save hidden states for each exit layer
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0 is embeddings, index i is output of layer i-1
    for layer in exit_layers:
        # hidden_states[layer] = output of layer (layer-1), before that layer
        # Actually in Qwen2.5: hidden_states[0] = embeddings, hidden_states[i] = output after layer i-1
        # So hidden_states[layer] = output after the (layer-1)-th decoder layer
        # = input to the layer-th decoder layer
        if layer < len(outputs.hidden_states):
            result[f'hidden_state_layer{layer}'] = outputs.hidden_states[layer].cpu()[0]

    # Final hidden state (before norm) - this is the output of the last decoder layer
    # hidden_states[-1] is the final normed output, hidden_states[-2] would be pre-norm
    # But Qwen2.5 model returns normed hidden_states as last_hidden_state
    # The actual pre-norm hidden state is hidden_states[num_layers] (0-indexed: after all layers, before norm)
    # Actually, when output_hidden_states=True in the Qwen2_5_VLModel.forward():
    # all_hidden_states includes states BEFORE each layer and the FINAL normed state
    # So hidden_states[-1] is the normed final state
    # We want the un-normed final state, which is hidden_states[-2] if the model appends normed at end
    # Let's use hidden_states[-1] since that's what the lm_head operates on
    result['hidden_state'] = outputs.hidden_states[-1].cpu()[0]

    return result


def main():
    args = parse_args()
    exit_layers = [int(x) for x in args.exit_layers.split(',')]

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).eval().to('cuda:0')

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)

    end = args.end if args.end is not None else len(data)
    data = data[args.start:end]
    print(f"Processing {len(data)} examples (from index {args.start} to {end})")

    os.makedirs(args.output_dir, exist_ok=True)

    num_processed = 0
    for i, example in enumerate(tqdm(data)):
        result = process_single_example(
            model, processor, example, exit_layers,
            args.max_seq_len, args.frame_interval,
        )
        if result is None:
            continue

        # Save as pytorch checkpoint
        save_path = os.path.join(args.output_dir, f'data_{args.start + i}.ckpt')
        torch.save(result, save_path)
        num_processed += 1

    print(f"Done! Processed {num_processed}/{len(data)} examples.")
    print(f"Saved to {args.output_dir}")
    print(f"Exit layers saved: {exit_layers}")


if __name__ == '__main__':
    main()
