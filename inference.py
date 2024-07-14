import os
import json
import torch
import socket
import deepspeed
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from utils import *
from tqdm import tqdm
from functools import partial
from datetime import datetime
from datetime import timedelta
from typing import Optional, Set
from transformers import set_seed
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments
)
from dataset import SftDataset


def main(args):
    set_seed(args.seed)

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )
    policy.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  

    eval_dataset = SftDataset(
        dataset_names=args.dataset_names.split(","), 
        split="test", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        max_prompt_length=args.max_prompt_length,
        n_samples = args.n_samples, 
        human_prefix=args.human_prefix, 
        human_suffix=args.human_suffix, 
        assistant_prefix=args.assistant_prefix, 
        assistant_suffix=args.assistant_suffix 
        )

    eval_iterator = DataLoader(eval_dataset, args.batch_size, shuffle=False, num_workers=8, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    
    # if loading from pt file
    # state_dict = torch.load(os.path.join(config.cache_dir, config.saved_policy), map_location='cpu')
    # step, metrics = state_dict['step_idx'], state_dict['metrics']
    # print(f'loading pre-trained weights for policy at step {step} from {config.saved_policy} with metrics {json.dumps(metrics, indent=2)}')
    # policy.load_state_dict(state_dict['state'])

    inf_config = {
        "replace_with_kernel_inject": False,
        "dtype": torch.bfloat16,
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": 8},
        'min_out_tokens': 1,
    }
    model = deepspeed.init_inference(model=policy, config=inf_config)
    all_policy_samples, all_prompts, all_chosen, all_original_prompts = [], [], [], []
    samples = []
    for b_idx, batch in tqdm(enumerate(eval_iterator), desc="Generating", total=len(eval_iterator)):
        with torch.no_grad():
            outputs = model.module.generate(
                    batch['prompt_input_ids'].to(torch.cuda.current_device()),
                    attention_mask=batch['prompt_attention_mask'].to(torch.cuda.current_device()),
                    max_length=args.max_length,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    top_p=args.top_p,
                )
            if accelerator.is_main_process:
                policy_samples = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_policy_samples.extend(policy_samples)

                chosen_samples = batch['chosen_text']
                all_chosen.extend(chosen_samples)
                all_prompts.extend(batch['prompt_text'])
                all_original_prompts.extend(batch['original_prompt'])
                
                for i in range(len(all_prompts)):
                    samples.append({
                        'prompt' : all_prompts[i],
                        'chosen' : all_chosen[i],
                        'policy' : all_policy_samples[i][len(all_prompts[i]):],
                        'original_prompt' : all_original_prompts[i],
                    })
                    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True) 
        fn = os.path.join(args.output_dir, f'{args.exp_name}.json')
        json.dump({
            'sampled_at' : str(datetime.now()),
            'samples' : samples,
        }, open(fn, 'w'), indent=2)

if __name__ == '__main__':

    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
        dataset_names: Optional[str] = field(default="hh", metadata={"help": "the dataset name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        batch_size: Optional[int] = field(default=4, metadata={"help": "bz"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        n_samples: Optional[int] = field(default=1, metadata={"help": "Number of samples; negative means all"})
        top_p: Optional[float] = field(default=0.95, metadata={"help": "top p"})
        output_dir: Optional[str] = field(default="./samples", metadata={"help": "directory"})
        exp_name: Optional[str] = field(default="sft", metadata={"help": "file name"})
        human_prefix: Optional[str] = field(default="\n<|user|>\n", metadata={"help": "mark of user talk"})
        assistant_prefix: Optional[str] = field(default="\n<|assistant|>\n", metadata={"help": "mark of model talk"})
        human_suffix: Optional[str] = field(default="", metadata={"help": "mark of user talk end"})
        assistant_suffix: Optional[str] = field(default="", metadata={"help": "mark of model talk end"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) # wait for processing upto 5hrs
    main(args)
