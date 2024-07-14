import tqdm
import torch
import random
import datasets
from utils import *
import pandas as pd
from typing import List
from collections import defaultdict
from torch.utils.data import Dataset


def get_alpacaeval(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    """
    Load the AlpacaEval dataset (for evaluation only) and convert it into to a Dataset.

    Args:
        - split: must be 'test'; otherwise error will be thrown
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        data list
    """
    if split == 'test':
        split = 'eval'
    else:
        raise ValueError('alpacaeval is only for evaluation')

    dataset = datasets.load_dataset('tatsu-lab/alpaca_eval', split=split)
    dataset = tqdm.tqdm(dataset, desc='Processing AlpacaEval')

    res = []
    for row in dataset:
        item = {}
        prompt = human_prefix + row['instruction'] + human_suffix + assistant_prefix
        item["prompt"] = prompt
        item["winer_generation"] = row['output'] + assistant_suffix
        item["dataset_name"] = row['dataset']
        # keep original prompt so that it can be dumped into a JSON file before running the alpacaeval command
        item["original_prompt"] = row['instruction']
        res.append(item)

    return res


def get_shp(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    """
    Load the Stanford Human Preferences dataset from Huggingface and convert it into to a Dataset.

    We filter preference pairs to only keep pairs where the score ratio is at least 2 (as in original SHP).
    For this dataset, the SFT text is the first response in SHP for a given prompt. 
    This is because the globally best response cannot be inferred from SHP, but all responses are a good option because they have a positive score.

    As recommended in the SteamSHPs' (reward models) data cards:
        Maximum number of pairs per prompt is 5 (in the training data, to avoid overfitting).
        Minimum score ratio of preferred to dispreferred response is 2

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        data list
    """
    MIN_SCORE_RATIO = 2
    MAX_PAIRS_PER_PROMPT = 5

    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split)
    dataset = tqdm.tqdm(dataset, desc='Processing SHP')

    prompt_to_items = defaultdict(list)
    
    for row in dataset:
        item = {}

        prompt = human_prefix + row['history'] + human_suffix + assistant_prefix
        responses = [row['human_ref_A'] + assistant_suffix, row['human_ref_B'] + assistant_suffix]
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])

        if score_ratio < MIN_SCORE_RATIO and split == 'train':
            continue

        item["original_prompt"] = prompt
        item["prompt"] = remove_extra_spaces(prompt)
        if row['labels'] == 1:
            item["winer_generation"], item["loser_generation"] = remove_extra_spaces(responses[0]), remove_extra_spaces(responses[1])
        else:
            item["winer_generation"], item["loser_generation"] = remove_extra_spaces(responses[1]), remove_extra_spaces(responses[0])
        item["truncation_mode"] = 'keep_start'
        item["dataset_name"] = "shp"

        prompt_to_items[remove_extra_spaces(prompt)].append(item)

    # prevent over-fitting
    if split == 'train':
        for prompt in prompt_to_items:
            prompt_to_items[prompt] = random.sample(prompt_to_items[prompt], min(MAX_PAIRS_PER_PROMPT, len(prompt_to_items[prompt])))
   
    res = []
    for prompt in prompt_to_items:
        res.extend(prompt_to_items[prompt])

    return res


def get_hh(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str, only_helpful = False, only_harmless = False):
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.
    
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - only_helpful: only the helpfulness data
        - only_harmless: only the harmlessness data

    Returns:   
        data list
    """
    if only_helpful:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="helpful-base")
    elif only_harmless:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, data_dir="harmless-base")
    else:
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split)
        
    dataset = tqdm.tqdm(dataset, desc='Processing HH')

    def split_prompt_and_responses(ex):
        search_term = '\n\nAssistant: '
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    res = []
    for row in dataset:
        item = {}

        prompt, chosen, rejected = split_prompt_and_responses(row)
        # strip trailing spaces to avoid tokenization issues
        chunks = []
        # turn doesn't always start with \n\n so watch out
        for chunk in re.split(r'\s*(Human:|Assistant:)\s+', prompt): 
            if chunk.startswith('Human'):
                chunk = re.sub(r'\s*Human:\s*', human_prefix, chunk) + human_suffix
            elif chunk.startswith('Assistant'):
                chunk = re.sub(r'\s*Assistant:\s*', assistant_prefix, chunk) + assistant_suffix
            else:
                pass

            if chunk != '':
                chunks.append(chunk)

        prompt = ''.join(chunks)
        responses = [chosen + assistant_suffix, rejected + assistant_suffix]

        item["original_prompt"] = prompt
        item["prompt"] = remove_extra_spaces(prompt)
        item["winer_generation"], item["loser_generation"] = remove_extra_spaces(responses[0]), remove_extra_spaces(responses[1])
        item["truncation_mode"] = 'keep_end'

        if only_helpful:
            item["dataset_name"] = 'hh_helpful'
        elif only_harmless:
            item["dataset_name"] = 'hh_harmless'
        else:
            item["dataset_name"] = 'hh'
        res.append(item)
        
    return res


def get_hh_helpful(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    return get_hh(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix, only_helpful=True)


def get_hh_harmless(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    return get_hh(split, human_prefix, human_suffix, assistant_prefix, assistant_suffix, only_harmless=True)


def get_oasst(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    """
    Load the Open Assistant dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    OASST is a dataset of ranked responses (not just pairwise), but since we are working with losses that expect paired preferences, 
    turn a ranking (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).
    
    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        data list
    """
    dataset = datasets.load_dataset('OpenAssistant/oasst1', split=('validation' if split == 'test' else 'train'))
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    message_indexed_df = pd.DataFrame(dataset).set_index('message_id')
    parent_indexed_df = pd.DataFrame(dataset).set_index('parent_id')

    def get_path_to_root(node: pd.Series):
        if node['parent_id'] is None:
            return [node]
        else:
            parent = message_indexed_df.loc[node['parent_id']]
            return [node] + get_path_to_root(parent)
    
    def turn_path_to_prompt(path: List[pd.Series]):
        prompt = []
        while path != []:
            node = path.pop() # earlier messages are at end of list
            prefix = assistant_prefix if node['role'] == 'assistant' else human_prefix
            suffix = assistant_suffix if node['role'] == 'assistant' else human_suffix
            prompt.append(prefix + node['text'] + suffix)
        
        prompt.append(assistant_prefix)
        return "".join(prompt)

    res = []
    for row in (tqdm.tqdm(dataset, desc='Processing OASST')):
        item = {}
        if row['rank'] == 0 or row['rank'] is None:
            continue

        try:
            sibling_df = parent_indexed_df.loc[row['parent_id']]
            next_best_sibling = sibling_df[sibling_df['rank'] == (row['rank'] - 1)].iloc[0]
            path_to_root = get_path_to_root(message_indexed_df.loc[next_best_sibling['message_id']])
        except KeyError:
            continue
        except IndexError:
            continue

        prompt = turn_path_to_prompt(path_to_root[1:])
        responses = [next_best_sibling['text'] + assistant_suffix, row['text'] + assistant_suffix]
       
        item["original_prompt"] = prompt
        item["prompt"] = remove_extra_spaces(prompt)
        item["winer_generation"], item["loser_generation"] = remove_extra_spaces(responses[0]), remove_extra_spaces(responses[1])
        item["truncation_mode"] = 'keep_end'
        item["dataset_name"] = 'oasst'

        res.append(item)
    
    return res


def get_ultrabin(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str):
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        data list
    """
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    else:
        raise ValueError()
    
    dataset = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized', split=split)
    dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')

    res = []
    for row in dataset:
        item = {}
        prompt = human_prefix + row['prompt'] + human_suffix + assistant_prefix
        responses = [row['chosen'][-1]['content'] + assistant_suffix, row['rejected'][-1]['content'] + assistant_suffix]

        item["original_prompt"] = prompt
        item["prompt"] = remove_extra_spaces(prompt)
        item["winer_generation"], item["loser_generation"] = remove_extra_spaces(responses[0]), remove_extra_spaces(responses[1])
        item["truncation_mode"] = 'keep_start'
        item["dataset_name"] = "ultrabin"

        res.append(item)

    return res

class SftDataset(Dataset):
    def __init__(self, dataset_names, split, tokenizer, max_length, max_prompt_length, n_samples, human_prefix, human_suffix, assistant_prefix, assistant_suffix):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.human_prefix = human_prefix
        self.human_suffix = human_suffix
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix
        self.n_samples = n_samples

        self.raw_data = []
        for name in dataset_names:
            temp_data = globals()[f"get_{name}"](split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
            self.raw_data.extend(temp_data)
        random.shuffle(self.raw_data)

        self.data = []
        if n_samples > 0:
            self.raw_data = self.raw_data[:self.n_samples]
        for item in self.raw_data:
            prompt_input_ids, generation_labels = self.tokenize_element(item["prompt"], item["winer_generation"], item["truncation_mode"])
            full_labels = [-100] * len(prompt_input_ids) + generation_labels
            full_input_ids = prompt_input_ids + generation_labels
            self.data.append({"input_ids": torch.LongTensor(full_input_ids), "labels": torch.LongTensor(full_labels), "prompt_input_ids": prompt_input_ids, "prompt": item["prompt"], "winer_generation": item["winer_generation"], "original_prompt": item["original_prompt"]})

    
    def tokenize_element(self, prompt: str, generation: str, truncation_mode: str):

        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]
  
        return prompt_token_ids, generation_token_ids


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_dpodataset(dataset_names, split, n_samples, human_prefix, human_suffix, assistant_prefix, assistant_suffix):
        raw_data = []
        for name in dataset_names:
            temp_data = globals()[f"get_{name}"](split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
            raw_data.extend(temp_data)
        random.shuffle(raw_data)
        
        truncation_mode=raw_data[0]["truncation_mode"]

        data = []
        if n_samples > 0:
            raw_data = raw_data[:n_samples]
        for item in raw_data:
            data.append({"prompt": item["prompt"], "chosen": item["winer_generation"], "rejected": item["loser_generation"]})

        return datasets.Dataset.from_list(data), truncation_mode

# class DpoDataset(datasets.Dataset):
#     def __init__(self, dataset_names, split, n_samples, human_prefix, human_suffix, assistant_prefix, assistant_suffix):

#         self.human_prefix = human_prefix
#         self.human_suffix = human_suffix
#         self.assistant_prefix = assistant_prefix
#         self.assistant_suffix = assistant_suffix
#         self.n_samples = n_samples

#         self.raw_data = []
#         for name in dataset_names:
#             temp_data = globals()[f"get_{name}"](split, human_prefix, human_suffix, assistant_prefix, assistant_suffix)
#             self.raw_data.extend(temp_data)
#         random.shuffle(self.raw_data)
        
#         self.truncation_mode=self.raw_data[0]["truncation_mode"]

#         self.data = []
#         if n_samples > 0:
#             self.raw_data = self.raw_data[:self.n_samples]
#         for item in self.raw_data:
#             self.data.append({"prompt": item["prompt"], "chosen": item["winer_generation"], "rejected": item["loser_generation"]})


#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

# Test
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    ds = SftDataset(dataset_names=["hh"], split="train", tokenizer=tokenizer, max_length=2048, max_prompt_length=1024, human_prefix="", human_suffix="", assistant_prefix="", assistant_suffix="")