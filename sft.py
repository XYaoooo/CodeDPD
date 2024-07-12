# Fine-Tune Llama2-7b on SE paired dataset
import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field, asdict
import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments
)

from trl import SFTConfig, SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset
from dataset import SftDataset

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def main(args):
    set_seed(args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=True,
    )
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

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=False,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        max_seq_length=args.max_length,
        run_name="SFT_Exp",
        report_to="none",
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
        dataset_names: Optional[str] = field(default="hh", metadata={"help": "the dataset name"})
        max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt lengthg"})
        max_length: Optional[int] = field(default=2048, metadata={"help": "the max sequence length"})
        batch_size: Optional[int] = field(default=4, metadata={"help": "bz"})
        learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "learning rate"})
        lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "learning rate decay"})
        warmup_ratio: Optional[float] = field(default=0.05, metadata={"help": "warm up"})
        weight_decay: Optional[float] = field(default=0.01, metadata={"help": "weight decay"})
        seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
        bf16: Optional[bool] = field(default=True, metadata={"help": "bf 16"})
        gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "gradient accumulation steps"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "None"})
        output_dir: Optional[str] = field(default="./checkpoints", metadata={"help": "directory"})
        num_train_epochs: Optional[float] = field(default=1, metadata={"help": "training epoches"})
        human_prefix: Optional[str] = field(default="\n<|user|>\n", metadata={"help": "mark of user talk"})
        assistant_prefix: Optional[str] = field(default="\n<|assistant|>\n", metadata={"help": "mark of model talk"})
        human_suffix: Optional[str] = field(default="", metadata={"help": "mark of user talk end"})
        assistant_suffix: Optional[str] = field(default="", metadata={"help": "mark of model talk end"})


    parser = HfArgumentParser(ScriptArguments)
    (args, ) = parser.parse_args_into_dataclasses()
    

    main(args)

    
