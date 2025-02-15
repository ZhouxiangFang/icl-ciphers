from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import argparse

llama8b_id = "meta-llama/Meta-Llama-3.1-8B"
llama8b_ins_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama70b_id = "meta-llama/Meta-Llama-3.1-70B"
qwen7b_id = 'Qwen/Qwen2.5-7B' 
olmo7b_id = 'allenai/OLMo-7B-0724-hf' 
gemma9b_id = 'google/gemma-2-9b' 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="llama3.1-8b",
        type=str,
        help="model name",
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, args.hf_token)

    ds = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
    id_count = {id:0 for id in range(len(tokenizer))}

    batch = 64

    for i in tqdm(range(0, len(ds), batch), total=len(ds)/batch):
        end = min(i+batch, len(ds))
        texts = ds[i:end]['text']
        ids = tokenizer.batch_encode_plus(texts, add_special_tokens=False)
        for sentence in ids['input_ids']:
            for index in sentence:
                id_count[index] += 1

    with open(f'tf_wikipedia_{args.model}.json', 'w') as json_file:
        json.dump(id_count, json_file)
        
if __name__ == "__main__":
    main()