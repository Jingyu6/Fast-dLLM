import argparse
import time

import torch
from generate import (generate, generate_with_dual_cache,
                      generate_with_prefix_cache)
from model.modeling_llada import LLaDAModelLM
from tqdm import tqdm
from transformers import AutoTokenizer


def single_inference(
    model, tokenizer, 
    use_cache, if_cache_position, 
    prompt, steps, gen_length, block_length, threshold
):
    if use_cache:
        if if_cache_position:
            out, nfe = generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)
        else:
            out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)
    else:
        out, nfe = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence', threshold=threshold)

    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    return answer, nfe


def main(args):
    # model init
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    m = [{"role": "user", "content": args.prompt}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(user_input)['input_ids']
    prompt = torch.tensor(input_ids).to(device).unsqueeze(0)

    torch.cuda.synchronize()
    for _ in tqdm(range(args.warmup), "Warmup..."):
        single_inference(
            model, tokenizer, 
            args.use_cache, args.if_cache_position, 
            prompt, args.steps, args.gen_length, args.block_size, args.threshold
        )

    total_gen_tokens = 0
    total_nfe = 0

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in tqdm(range(args.iteration), "Real profiling..."):
        answer, nfe = single_inference(
            model, tokenizer, 
            args.use_cache, args.if_cache_position, 
            prompt, args.steps, args.gen_length, args.block_size, args.threshold
        )
        total_gen_tokens += len(tokenizer.encode(answer))
        total_nfe += nfe

        if args.print_result:
            print("==========================")
            print(answer)
            print("==========================")

        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time

    print("==============================================")
    print(f"Total time elapsed: {total_time:.5f}s")
    print(f"Gen TPS: {total_gen_tokens / total_time:.2f} tokens / second")
    print(f"nfe per second: {nfe / total_time:.2f} steps / second")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--if_cache_position", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=8)
    parser.add_argument("--print_result", action="store_true")

    args = parser.parse_args()
    main(args)
