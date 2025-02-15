import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import json

llama8b_id = "meta-llama/Meta-Llama-3.1-8B"
llama8b_ins_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama70b_id = "meta-llama/Meta-Llama-3.1-70B"
qwen7b_id = 'Qwen/Qwen2.5-7B' 
olmo7b_id = 'allenai/OLMo-7B-0724-hf' 
gemma9b_id = 'google/gemma-2-9b' 

def text2ids(question, options, args, tokenizer):
    if 'hellaswag' in args.dataset:
        text = question + " " + options
    elif 'winogrande' in args.dataset:
        text = " " + question
    else:
        text = question
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='sst_2',
        type=str,
        help="dataset: sst_2, amazon, hellaswag, winogrande",
    )
    parser.add_argument(
        "--model",
        default="llama3.1-8b",
        type=str,
        help="model name: llama3.1-8b",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="huggingface token",
    )

    args = parser.parse_args()

    if args.model == 'llama3.1-8b':
        model_id = llama8b_id
    elif args.model == 'llama3.1-8b-ins':
        model_id = llama8b_ins_id
    elif args.model == 'llama3.1-70b':
        model_id = llama70b_id
    elif args.model == 'qwen7b':
        model_id = qwen7b_id
    elif args.model == 'olmo7b':
        model_id = olmo7b_id
    elif args.model == 'gemma9b':
        model_id = gemma9b_id
    else:
        raise ValueError(f'Model {args.model} unimplemented')

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)

    df = pd.read_json(f'dataset/{args.dataset}.json',lines=True)
    df_demos = pd.read_json(f'dataset/{args.dataset}_demos.json',lines=True)

    my_set = set()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if 'hellaswag' in args.dataset:
            ids = text2ids(row['question'], row['options'], args, tokenizer)
        elif 'winogrande' in args.dataset:
            ids = text2ids(row['question'], None, args, tokenizer)
        else:
            ids = text2ids(row['input'], None, args, tokenizer)
        my_set.update(ids)

    if 'hellaswag' in args.dataset:
        df_demos['ids'] = df_demos.apply(lambda row: text2ids(row['question'], row['options'], args, tokenizer), axis=1)
    elif 'winogrande' in args.dataset:
        df_demos['ids'] = df_demos.apply(lambda row: text2ids(row['question'], None, args, tokenizer), axis=1)
    else:
        df_demos['ids'] = df_demos.apply(lambda row: text2ids(row['input'], None, args, tokenizer), axis=1)

    demo_dict = {}
    for token_id in tqdm(my_set, total=len(my_set)):
        demo_indexs = list(df_demos[df_demos['ids'].apply(lambda x: token_id in x)].index)
        demo_dict[token_id] = demo_indexs

    with open(f'dataset/{args.dataset}_demo_dict_{args.model}.json', 'w') as json_file:
        json.dump(demo_dict, json_file)
        

if __name__ == "__main__":
    main()
