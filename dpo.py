# Fine-Tune Llama2-7b on SE paired dataset
import os
import torch
from accelerate import Accelerator
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field, asdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments
)
from trl import DPOConfig, DPOTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset
from dataset import *


def main(args):
    set_seed(args.seed)

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset, truncation_mode = get_dpodataset(dataset_names=args.dataset_names.split(","),
        split="train", 
        n_samples = args.n_samples, 
        human_prefix=args.human_prefix, 
        human_suffix=args.human_suffix, 
        assistant_prefix=args.assistant_prefix, 
        assistant_suffix=args.assistant_suffix 
        )

    training_args = DPOConfig(
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        dataloader_drop_last=False,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reetrant),
        bf16=args.bf16,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        truncation_mode=truncation_mode,
        remove_unused_columns=False,
        run_name="DPO_Exp",
        report_to="none"
    )

    trainer = DPOTrainer(
        policy_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)



if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the policy model name"})
        ref_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the reference model name"})
        dataset_names: Optional[str] = field(default="hh", metadata={"help": "the dataset name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        batch_size: Optional[int] = field(default=2, metadata={"help": "bz"})
        learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "learning rate"})
        beta: Optional[float] = field(default=0.1, metadata={"help": "beta"})
        lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "learning rate decay"})
        warmup_ratio: Optional[float] = field(default=0.01, metadata={"help": "warm up"})
        weight_decay: Optional[float] = field(default=0.00, metadata={"help": "weight decay"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "gradient accumulation steps"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "None"})
        gradient_checkpointing_use_reetrant: Optional[bool] = field(default=False, metadata={"help": "None"})
        n_samples: Optional[int] = field(default=100, metadata={"help": "number of sample; negative means all"})
        output_dir: Optional[str] = field(default="./DPO_checkpoints", metadata={"help": "directory"})
        num_train_epochs: Optional[float] = field(default=1, metadata={"help": "training epoches"})
        human_prefix: Optional[str] = field(default="\n<|user|>\n", metadata={"help": "mark of user talk"})
        assistant_prefix: Optional[str] = field(default="\n<|assistant|>\n", metadata={"help": "mark of model talk"})
        human_suffix: Optional[str] = field(default="", metadata={"help": "mark of user talk end"})
        assistant_suffix: Optional[str] = field(default="", metadata={"help": "mark of model talk end"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    

    main(args)

    
