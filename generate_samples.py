# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import nltk
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers, set_random_seed
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import GPT2Model
from utils import print_rank_0

from flask import Flask, request, abort

USE_TORCH_DDP = True


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      max_memory_length=args.mem_length,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False,
                      relative_encoding=args.transformer_xl)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if hasattr(args, "deepspeed") and args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        from model import PyTorchDistributedDataParallel as DDP
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        from model import DistributedDataParallel as DDP
        model = DDP(model)

    return model


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               transformer_xl=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if transformer_xl:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        loss_mask[data == eod_token] = 0.0
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
            print(f"Load model file {path}")
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir,
        'add_eop': args.hierarchical}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


def read_context(prompt, tokenizer, args):
    if mpu.get_model_parallel_rank() == 0:
        context_tokens = tokenizer.EncodeAsIds(prompt).tokenization
        context_length = len(context_tokens)
    else:
        context_length = 0

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    if mpu.get_model_parallel_rank() != 0:
        prompt = tokenizer.DecodeIds(context_tokens_tensor.tolist())
    return prompt, context_tokens_tensor, context_length


def sample_sequence(model, stop_words, tokenizer, context_tokens_tensor, context_length, args, device, mems=None):
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

    counter = 0
    if mems is None:
        mems = []

    while counter < args.out_seq_length:
        if counter == 0:
            logits, *mems = model(tokens, position_ids, attention_mask, *mems)
        else:
            index = context_length + counter
            new_position_ids = tokens.new_ones((1, 1)) * (index - 1)
            new_attention_mask = tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device, dtype=torch.float)
            logits, *mems = model(tokens[:, index - 1: index], new_position_ids, new_attention_mask, *mems)

        logits = logits[:, -1]
        logits /= args.temperature
        logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
        log_probs = F.softmax(logits, dim=-1)
        
        prev = torch.multinomial(log_probs, num_samples=1)[0]
        if prev == args.eod_token:
            break
        elif prev in stop_words and counter > args.min_out_seq_length:
            tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            break

        tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
        counter += 1

    output_tokens_list = tokens.view(-1).contiguous()
    return output_tokens_list, mems


def generate_samples(prompt, stop_words, model, tokenizer, args, device):
    model.eval()
    with torch.no_grad():
        torch.distributed.barrier(group=mpu.get_model_parallel_group())

        start_time = time.time()
        prompt, context_tokens_tensor, context_length = read_context(prompt, tokenizer, args)
        output_tokens_list, _ = sample_sequence(model, stop_words, tokenizer, context_tokens_tensor, context_length, args, device)
        decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
        trim_decode_tokens = decode_tokens[len(prompt):]

        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        print('Length range: {} - {}'.format(args.min_out_seq_length, args.out_seq_length))
        print('Prompt: {}'.format(prompt))
        print('Generation: {}'.format(trim_decode_tokens))
        print("Taken time {:.2f}".format(time.time() - start_time), flush=True)
        return trim_decode_tokens


def generate(request, model, tokenizer, args):
    print('Processing request: {}'.format(request.json))
    PROMPT_KEY = 'prompt'
    MAX_TOKENS_KEY = 'max_length'
    MIN_TOKENS_KEY = 'min_length'
    TEMPERATURE_KEY = 'temperature'
    TOP_K_KEY = 'top_k'
    TOP_P_KEY = 'top_p'
    STOP_WORDS_KEY = 'stop_words'

    DEFAULT_MAX_TOKEN = 1024
    DEFAULT_MIN_TOKEN = 32
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 100

    char_map = {
        ',': '，',
        '.': '。',
        ':': '：',
        '!': '！',
        '?': '？',
        '(': '（',
        ')': '）',
        '#': '\n',
    }

    if not request.json or not PROMPT_KEY in request.json:
            abort(400)

    prompt = request.json[PROMPT_KEY]
    stop_words = request.json[STOP_WORDS_KEY] if STOP_WORDS_KEY in request.json else []
    max_token = request.json[MAX_TOKENS_KEY] if MAX_TOKENS_KEY in request.json else DEFAULT_MAX_TOKEN
    min_token = request.json[MIN_TOKENS_KEY] if MIN_TOKENS_KEY in request.json else DEFAULT_MIN_TOKEN
    temperature = request.json[TEMPERATURE_KEY] if TEMPERATURE_KEY in request.json else DEFAULT_TEMPERATURE
    top_p = request.json[TOP_P_KEY] if TOP_P_KEY in request.json else DEFAULT_TOP_P
    top_k = request.json[TOP_K_KEY] if TOP_K_KEY in request.json else DEFAULT_TOP_K

    args.out_seq_length = max_token
    args.min_out_seq_length = min_token
    args.temperature = temperature
    args.top_k = top_k
    args.top_p = top_p

    stop_ids = [45225]
    # for word in stop_words:
    #     stop_ids += tokenizer.EncodeAsIds(word).tokenization

    prompt = prompt.replace('\n', '#')
    result = generate_samples(prompt, stop_ids, model, tokenizer, args, torch.cuda.current_device())
    for en, ch in char_map.items():
        result = result.replace(en, ch)

    return result, 200


def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    args.deepspeed = False
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # Start up generation service
    app = Flask(__name__)

    @app.route('/generation', methods=['POST'])
    def generation():
        return generate(request, model, tokenizer, args)

    app.run(host='0.0.0.0',port=5000)


if __name__ == "__main__":
    main()
