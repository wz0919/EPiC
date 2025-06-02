#!/usr/bin/env python3
import argparse
import torch
import os
import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer

def compute_prompt_embeddings(tokenizer, text_encoder, prompts, max_sequence_length=226, device=torch.device("cpu"), dtype=torch.float16):
    if isinstance(prompts, str):
        prompts = [prompts]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    ).to(device)

    all_files = sorted(os.listdir(args.caption_path))
    chunk = all_files[args.start_idx: args.end_idx]

    os.makedirs(args.output_path, exist_ok=True)

    for name in tqdm.tqdm(chunk, desc=f"GPU {args.gpu_id}"):
        with open(os.path.join(args.caption_path, name), 'r') as f:
            caption = f.read().strip()

        embeddings = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            caption,
            max_sequence_length=args.max_sequence_length,
            device=device,
            dtype=torch.bfloat16
        ).cpu()
        torch.save(embeddings, os.path.join(args.output_path, name.replace('.txt', '') + '.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU T5 prompt embedding")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--caption_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)
