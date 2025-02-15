import random
import copy
import pandas as pd
import argparse
import numpy as np
import os
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import warnings
import nltk

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

llama8b_id = "meta-llama/Meta-Llama-3.1-8B"
llama8b_ins_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama70b_id = "meta-llama/Meta-Llama-3.1-70B"
qwen7b_id = 'Qwen/Qwen2.5-7B' 
olmo7b_id = 'allenai/OLMo-7B-0724-hf' 
gemma9b_id = 'google/gemma-2-9b' 

def get_pids(tokenizer, preserve_ids, preserve_tokens):
    for token in preserve_tokens:
        id = tokenizer.encode(token, add_special_tokens=False)[0]
        preserve_ids.append(id)
        return preserve_ids

def shullfe_ids(old_ids, shuffle_rate):
    V = len(old_ids)
    shuffle_num = int(V * shuffle_rate)
    shuffle_index = random.sample(list(range(V)), shuffle_num)
    shuffle_id = [old_ids[idx] for idx in shuffle_index]
    random.shuffle(shuffle_id)
    new_ids = copy.deepcopy(old_ids)
    for i, idx in enumerate(shuffle_index):
        new_ids[idx] = shuffle_id[i]
    return new_ids

def shuffle_sub_ids(id2tokens, args):

    space_ids = []
    nonspace_ids = []
    for id, token in id2tokens.items():
        if args.space_symbol in token:
            space_ids.append(id)
        else:
            nonspace_ids.append(id)

    shuffle_space_ids = shullfe_ids(space_ids, args.shuffle_rate)
    shuffle_nonspace_ids = shullfe_ids(nonspace_ids, args.shuffle_rate)

    space_ids_mapping = {}
    nonspace_ids_mapping = {}

    for i in range(len(space_ids)):
        key = space_ids[i]
        value = shuffle_space_ids[i]
        space_ids_mapping[key] = value

    for i in range(len(nonspace_ids)):
        key = nonspace_ids[i]
        value = shuffle_nonspace_ids[i]
        nonspace_ids_mapping[key] = value

    submapping = {**space_ids_mapping, **nonspace_ids_mapping}
    return submapping

def get_ids_mapping(tokenizer, preserve_ids, preserve_tokens, tf, args):

    if args.zipfian>1:
        for k, v in tf.items():
            if v==0:
                preserve_ids.append(k)
    pids = get_pids(tokenizer=tokenizer, preserve_ids=preserve_ids, preserve_tokens=preserve_tokens)

    vocab = tokenizer.get_vocab()
    id2tokens = {v: k for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    # limited to noun
    if args.pos:
        valid_tags = ['NN', 'NNS']
        noun_token_ids = []
        for k, v in tf.items():
            if k not in pids:
                token = tokenizer.convert_ids_to_tokens(k)
                if 'Ġ' in token and token.strip().isalpha() and len(token.strip())>1:
                    pos_tag = nltk.pos_tag([token.strip()])[0][1]
                    if pos_tag in valid_tags:
                        noun_token_ids.append(k)
        # print(noun_token_ids)
        tf = {k:v for k, v in tf.items() if k not in pids and k in noun_token_ids}
    else:
        # tokens to be shuffled
        tf = {k:v for k, v in tf.items() if k not in pids}
    tf = dict(sorted(tf.items(), key=lambda item: item[1], reverse=True))

    submapping_list = []
    subdict_size = int(len(tf)/args.zipfian)
    for i in range(args.zipfian):
        start = i*subdict_size
        if i==args.zipfian-1:
            end = len(tf)
        else:
            end = (i+1)*subdict_size
        sub_id2tokens = {k: id2tokens[k] for k in list(tf)[start:end]}
        submapping = shuffle_sub_ids(sub_id2tokens, args)
        submapping_list.append(submapping)
       
    ids_mapping = {}
    for index, v in id2tokens.items():
        ids_mapping[index] = index
    for submapping in submapping_list:
        for k, v in submapping.items():
            ids_mapping[k] = v
    ids_mapping = dict(sorted(ids_mapping.items(), key=lambda item: item[0]))
    return ids_mapping


def bi_sub_text(text, tokenizer, ids_mapping, args, space=False):
    if space:
        text = ' ' + text
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(ids)):
        id = ids[i]
        ids[i] = ids_mapping[id]
    subbed_text = tokenizer.decode(ids, skip_special_tokens=True)
    return subbed_text

def nonbi_sub_text(text, tokenizer, ids_mapping, id2tokens, args, space=False):
    if space:
        text = ' ' + text
    space_ids = []
    nonspace_ids = []
    for id, mapped_id in ids_mapping.items():
        if id != mapped_id:
            if args.space_symbol in id2tokens[id]:
                space_ids.append(id)
            else:
                nonspace_ids.append(id)

    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(ids)):
        id = ids[i]
        if id != ids_mapping[id]:
            token = id2tokens[id]
            if args.space_symbol in token:
                ids[i] = random.choice(space_ids)
            else:
                ids[i] = random.choice(nonspace_ids)
    subbed_text = tokenizer.decode(ids, skip_special_tokens=True)
    return subbed_text

def sample_demos(row, demo_dict, df_demos, tokenizer, ids_mapping, args):
    if args.sampling == 'non-priority':
        demo_indexs = random.sample(list(df_demos.index), args.fewshot)
    elif args.sampling == 'priority':
        if 'hellaswag' in args.dataset:
            ori_question = row['question'] + ' '+ row['options']
        elif 'winogrande' in args.dataset:
            ori_question = row['question']
        else:
            ori_question = row['input']
        ids = tokenizer.encode(ori_question, add_special_tokens=False)
        demo_indexs = []
        for token_id in ids:
            if ids_mapping[token_id]!=token_id and token_id in demo_dict and demo_dict[token_id]!=[]:
                demo_indexs.append(random.choice(demo_dict[token_id]))
        demo_indexs = list(set(demo_indexs))
        if len(demo_indexs)>=args.fewshot:
            demo_indexs = random.sample(demo_indexs, args.fewshot)
        else:
            remaining_indices = df_demos.index.difference(demo_indexs)
            indices = np.random.choice(remaining_indices, args.fewshot-len(demo_indexs), replace=False).tolist()
            demo_indexs = demo_indexs + indices

    assert len(demo_indexs) == args.fewshot
    random.shuffle(demo_indexs)
    return demo_indexs

def call_llm(tokenizer, model, fewshot_df, test_row, args):
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("Input"),
    tokenizer.convert_tokens_to_ids("Question")
    ]
    messages = ''
    if 'hellaswag' in args.dataset or 'winogrande' in args.dataset:
        for idx, row in fewshot_df.iterrows():
            messages += 'Question:{}\n'.format(row['subbed_question'])
            messages += 'Options:\n{}\n'.format(row['subbed_options'])
            messages += 'Answer: ({})\n\n'.format(row['answer'])

        messages += 'Question:{}\n'.format(test_row['subbed_question'])
        messages += 'Options:{}\n'.format(test_row['subbed_options'])
    else:
        for idx, row in fewshot_df.iterrows():
            messages += 'Input: {}\n'.format(row['subbed_input'])
            messages += 'Output: {}\n'.format(row['answer'])
        messages += f'''Input: {test_row['subbed_input']}\n'''
    # print(messages)
    encodeds = tokenizer.encode(messages,return_tensors="pt")
    model_inputs = encodeds.to(args.device)
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, max_new_tokens=10, eos_token_id = terminators, do_sample=True, temperature=0.001)
    decoded = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)
    output = decoded[0]

    return output

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
        help="model name",
    )
    parser.add_argument(
        "--fewshot",
        default="5",
        type=int,
        help="the numder of demonstration in in-context learning",
    )
    parser.add_argument(
        "--zipfian",
        default="1",
        type=int,
        help="zifpian shuffling",
    )
    parser.add_argument(
        "--sub",
        default="bijective",
        type=str,
        help="substitution strategy: bijective or non-bijective",
    )
    parser.add_argument(
        "--sampling",
        default="priority",
        type=str,
        help="sampling strategy: priority or nnon-priority",
    )
    parser.add_argument(
        "--result_dir",
        default="./",
        type=str,
        help="directory to save the results",
    )
    parser.add_argument(
        "--min",
        default="0.0",
        type=float,
        help="min shuffle rate",
    )
    parser.add_argument(
        "--max",
        default="1.0",
        type=float,
        help="max shuffle rate",
    )
    parser.add_argument(
        '--save', 
        default=False,
        action='store_true', 
        help="whether to save the output file"
    )
    parser.add_argument(
        '--pos', 
        default=False,
        action='store_true', 
        help="part of speech"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        help="huggingface token",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'{args.zipfian} Zipfian Shuffling' if args.zipfian>1 else 'Non-zipfian Shuffling')
    print(f'sampling strategy: {args.sampling}')
    print(f'substitution strategy: {args.sub}')
    print(f'few shot: {args.fewshot}')
    print(f'shuffle rate min: {args.min}')
    print(f'shuffle rate max: {args.max}')
    print(f'Part of Speech' if args.pos else 'Non Part of Speech')

    os.makedirs('ids_mapping', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if 'sst' in args.dataset or 'amazon' in args.dataset:
        valid_options = ['positive', 'negative']
    elif 'hellaswag' in args.dataset:
        valid_options = ['1', '2', '3', '4']
    elif 'winogrande' in args.dataset:
        valid_options = ['1', '2']
    else:
        raise ValueError('Dataset doesn\'t exist')
    print(f'valid options: {valid_options}')

    if args.model == 'llama3.1-8b':
        model_id = llama8b_id
        args.space_symbol = 'Ġ'
        preserve_ids = list(range(0,256)) + list(range(128000, 128256))
        preserve_tokens = [' ', 'Ġ_', '_','Ġ(', 'Ġ_ ','\n']
    elif args.model == 'llama3.1-8b-ins':
        model_id = llama8b_ins_id
        args.space_symbol = 'Ġ'
        preserve_ids = list(range(0,256)) + list(range(128000, 128256))
        preserve_tokens = [' ', 'Ġ_', '_','Ġ(', 'Ġ_ ','\n']
    elif args.model == 'llama3.1-70b':
        model_id = llama70b_id
        args.space_symbol = 'Ġ'
        preserve_ids = list(range(0,256)) + list(range(128000, 128256))
        preserve_tokens = [' ', 'Ġ_', '_','Ġ(', 'Ġ_ ','\n']
    elif args.model == 'qwen7b':
        model_id = qwen7b_id
        args.space_symbol = 'Ġ'
        preserve_ids = list(range(0,256)) + list(range(151643, 151665))
        preserve_tokens = [' ', 'Ġ_', '_','Ġ(', 'Ġ_ ','\n']
    elif args.model == 'olmo7b':
        model_id = olmo7b_id
        args.space_symbol = 'Ġ'
        preserve_ids = list(range(0, 245)) + list(range(50254, 50280))
        preserve_tokens = [' ', 'Ġ_', '_','Ġ(', 'Ġ_ ','\n']
    elif args.model == 'gemma9b':
        model_id = gemma9b_id
        args.space_symbol = '▁'
        preserve_ids = list(range(0, 473)) + list(range(255968, 256000))
        preserve_tokens = [' ', ' _', '_',' (', ' _ ','\n']
    else:
        raise ValueError(f'Model {args.model} unimplemented')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.hf_token
    )

    step = 0.1
    shuffle_rates = []
    current = args.min
    while current <= args.max:
        shuffle_rates.append(round(current, 1))
        current += step
    runs = [1,2,3]
    print("Start ICL")
    for rate in tqdm(shuffle_rates, total=len(shuffle_rates)):
        for run in runs:
            args.shuffle_rate = rate
            args.run = run

            df = pd.read_json(f'dataset/{args.dataset}.json', lines=True)
            df_demos = pd.read_json(f'dataset/{args.dataset}_demos.json', lines=True)
            if args.pos:
                mapping_file  = f'ids_mapping/{args.dataset}_{args.model}_{args.fewshot}_{args.shuffle_rate}_zf{args.zipfian}_run{args.run}_pos.json'
            else:
                mapping_file  = f'ids_mapping/{args.dataset}_{args.model}_{args.fewshot}_{args.shuffle_rate}_zf{args.zipfian}_run{args.run}.json'

            mapping_file = os.path.join(args.result_dir, mapping_file)

            if os.path.exists(mapping_file):
                print('ids_mapping already exists')
                with open(mapping_file, 'r') as f:
                    ids_mapping = json.load(f)
                ids_mapping = {int(k):int(v) for k,v in ids_mapping.items()}
            else:
                print('creating new ids_mapping')
                tf_file = f'tf_wikipedia_{args.model}.json'
                with open(tf_file, 'r') as f:
                    tf = json.load(f)
                tf = {int(k):int(v) for k,v in tf.items()}
                # shuffle the vocab and get the mapping of ids
                ids_mapping = get_ids_mapping(tokenizer, preserve_ids, preserve_tokens, tf, args)
                with open(mapping_file, 'w') as json_file:
                    json.dump(ids_mapping, json_file)
            if args.pos:
                file_name = f'results/{args.dataset}_{args.model}_zf{args.zipfian}_{args.sampling}_sampling_{args.sub}_sub_fs{args.fewshot}_sr{args.shuffle_rate}_run{args.run}_pos.json'
            else:
                file_name = f'results/{args.dataset}_{args.model}_zf{args.zipfian}_{args.sampling}_sampling_{args.sub}_sub_fs{args.fewshot}_sr{args.shuffle_rate}_run{args.run}.json'

            json_name = os.path.join(args.result_dir, file_name)

            if os.path.exists(json_name):
                continue
            
            if args.sub == 'non-bijective':
                vocab = tokenizer.get_vocab()
                id2tokens = {v: k for k, v in sorted(vocab.items(), key=lambda item: item[1])}
                if 'hellaswag' in args.dataset or 'winogrande' in args.dataset: # hellaswag and winogrande
                    df['subbed_question'] = df['question'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args, True))
                    df['subbed_options'] = df['options'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args))
                else: # sst-2 and amazon
                    df['subbed_input'] = df['input'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args))
            elif args.sub == 'bijective':
                if 'hellaswag' in args.dataset or 'winogrande' in args.dataset: # hellaswag and winogrande
                    df['subbed_question'] = df['question'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args, True))
                    df['subbed_options'] = df['options'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args))

                    df_demos['subbed_question'] = df_demos['question'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args, True))
                    df_demos['subbed_options'] = df_demos['options'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args))
                else: # sst-2 and amazon

                    df['subbed_input'] = df['input'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args))
                    df_demos['subbed_input'] = df_demos['input'].apply(bi_sub_text, args=(tokenizer, ids_mapping, args))
            else:
                raise ValueError('Invalid substitution strategy')
            
            outputs = []
            fewshot_indices = []
            predictions = []
            valid = 0
            correct = 0
            with open(f'dataset/{args.dataset}_demo_dict_{args.model}.json', 'r') as f:
                demo_dict = json.load(f)
            demo_dict = {int(k):v for k,v in demo_dict.items()}

            print('Start ICL')
            
            for idx, row in df.iterrows():
                demo_indexs = sample_demos(row, demo_dict, df_demos, tokenizer, ids_mapping, args)
                fewshot_indices.append(','.join([str(x) for x in demo_indexs]))
                fewshot_df = df_demos.loc[demo_indexs]
                if args.sub == 'non-bijective':
                    if 'hellaswag' in args.dataset or 'winogrande' in args.dataset: # hellaswag and winogrande
                        fewshot_df['subbed_question'] = fewshot_df['question'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args, True))
                        fewshot_df['subbed_options'] = fewshot_df['options'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args))
                    else:
                        fewshot_df['subbed_input'] = fewshot_df['input'].apply(nonbi_sub_text, args=(tokenizer, ids_mapping, id2tokens, args))
                answer = str(row['answer'])

                output = call_llm(tokenizer, model, fewshot_df, row, args)

                outputs.append(output)
                prediction = output
                for valid_option in valid_options:
                    if valid_option in output.lower():
                        prediction = valid_option
                        valid += 1
                        if prediction == answer:
                            correct += 1
                        break
                predictions.append(prediction)

            df['output'] = outputs
            df['prediction'] = predictions
            df['fewshot_indices'] = fewshot_indices

            if args.save:
                df.index.name = 'id'
                df = df.reset_index()
                json_dict = df.to_dict(orient='records')
                with open(json_name, 'w') as f:
                    json.dump(json_dict, f, indent=4)

    main()