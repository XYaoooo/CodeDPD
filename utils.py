import re
import torch

def remove_extra_spaces(str):
    clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
    cleaned_str = clean(str)
    return cleaned_str


def collate_fn(features, tokenizer):
    batch = {"prompt_input_ids": [], "prompt_attention_mask": [], "prompt_text": [], "chosen_text": [], "original_prompt": []}
    max_input_length = max(len(x["prompt_input_ids"]) for x in features)
    for feature in features:
        input_ids = feature["prompt_input_ids"]
        attention_mask = [1] * len(input_ids)
        input_ids = [tokenizer.pad_token_id] * (max_input_length - len(input_ids)) + input_ids
        attention_mask = [0] * (max_input_length - len(attention_mask)) + attention_mask
        batch["prompt_input_ids"].append(input_ids)
        batch["prompt_attention_mask"].append(attention_mask)
        batch["prompt_text"].append(feature["prompt"])
        batch["chosen_text"].append(feature["winer_generation"])
        batch["original_prompt"].append(feature["original_prompt"])
    batch["prompt_input_ids"] = torch.tensor(batch["prompt_input_ids"])
    batch["prompt_attention_mask"] = torch.tensor(batch["prompt_attention_mask"])
    
    return batch