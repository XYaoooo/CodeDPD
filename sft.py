# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from trl import SFTConfig, SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset
from dataset import SftDataset


def main(args):
    if args.group_by_length and training_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    set_seed(args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage = True,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset = SftDataset(
        dataset_names=args.dataset_names.split(","), 
        split="train", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        max_prompt_length=args.max_prompt_length, 
        human_prefix=args.human_prefix, 
        human_suffix=args.human_suffix, 
        assistant_prefix=args.assistant_prefix, 
        assistant_suffix=args.assistant_suffix 
        )


    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_batch_size=args.dataset_batch_size
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
        dataset_name: Optional[str] = field(default="hh,shp", metadata={"help": "the dataset name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        dataset_batch_size: Optional[int] = field(default=16, metadata={"help": "bz"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        epochs: Optional[int] = field(default=1, metadata={"help": "training epoches"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        human_prefix: Optional[str] = field(default="\n<|user|>\n", metadata={"help": "mark of user talk"})
        assistant_prefix: Optional[str] = field(default="\n<|assistant|>\n", metadata={"help": "mark of model talk"})
        human_suffix: Optional[str] = field(default="", metadata={"help": "mark of user talk end"})
        assistant_suffix: Optional[str] = field(default="", metadata={"help": "mark of model talk end"})


    parser = HfArgumentParser(ScriptArguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)

    main(args)

    
