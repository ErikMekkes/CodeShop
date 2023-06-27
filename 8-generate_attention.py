# this notebook trains a kmeans clustering using summed, scaled attention scores
# then we label new data using that learned clustering.

# training is done on random predictions from random files across all languages
# a single random head is taken from each layer in each predictions.
# lines per head = (n_tokens+1)*(n_tokens/2) = 1025*512 = 524800

import datasets
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import login
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

huggingface_token_filename = "./huggingface_token.txt"
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

device_type = 'cuda'
# device_type = 'cpu'


model_name = "NinedayWang/PolyCoder-0.4B"
model_shortname = "PolyCoder"
lens_name = "AISE-TUDelft/PolyCoder-lens"

tokenizer_name=model_name
# model specific
num_heads = 16
# as many as gpu can handle, ~2 per 5GB VRAM, ~10 for DHPC?
batch_size = 1
# we always want this size
input_size = 1024     # n x n attention scores


languages = ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"]
languages = ["Go"]

data = {}
data_lengths = []
for lang in languages:
    name = f"./Predictions/{model_shortname}/{model_shortname}CodeShop{lang}Predictions/"
    data[lang] = datasets.load_dataset(name, split="train")
    length = len(data[lang])
    print(f"{lang} samples: {length}", flush=True)
    data_lengths.append(length)
min_samples = np.min(data_lengths)
print(min_samples)

tokens_of_interest = {
    # "continue_tokens" : 0,
    # "return_tokens" : 0,
    # "else_tokens" : 1,
    # "for_tokens",
    # "if_tokens" : 2,
    # "assign_tokens": 3,
    # "key_tokens": 4,
    # "value_tokens": 5,
    # "sum_tokens",
    "start_tokens": 6,
    # "message_tokens": 7,
}

interest_inputs = {
    "else_tokens": [],
    "start_tokens": []
}
for lang in languages:
    interest_inputs[lang] = {}
    for t_name in tokens_of_interest.keys():
        interest_inputs[lang][t_name] = []

start_f = 4
end_f = 9
files = end_f-start_f
interesting_inputs = []
# for each token of interest
for lang in languages:
    name = f"./Interest/{lang}TokensOfInterest"
    interest = datasets.load_dataset(name, split="train")
    for t_name,t_index in tokens_of_interest.items():
        samples = interest[lang][t_index]
        for token in range(start_f,min(end_f,len(samples[0]))):
            sample_id = samples[0][token]
            sample_i = samples[3][token]
            token_i = samples[1][token]
            section = data[lang][sample_i]["tokenized_section"]
            input_ids = section[token_i-input_size:token_i]
            interest_inputs[lang][t_name].append(input_ids)
            interesting_inputs.append(input_ids)
            # = perfect input for prediction of token_type

def head_summed_scores(head):
    n_tokens = len(head)
    head_attentions = np.zeros(n_tokens)
    for n_1 in range(n_tokens):
        for n_2 in range(n_tokens):
            head_attentions[n_2] += head[n_1][n_2]
    return head_attentions

with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True
    )
    model.eval()
    device = torch.device(device_type)
    model.to(device)

    file_counter = 0
    split_counter = 0
    while file_counter < files:
        batch_inp = []
        for p_i in range(batch_size):
            # dataset = data['Java']
            # dataset_len = len(dataset)
            # sample = np.random.randint(0, dataset_len)
            # inp = dataset['tokenized_section'][sample]

            # pred_index = np.random.randint(0, len(inp)-input_size)
            # pred_tokens = torch.IntTensor(inp[pred_index:pred_index+input_size])
            pred_tokens = torch.IntTensor(interesting_inputs[file_counter])
            batch_inp.append(pred_tokens)
            file_counter += 1
            # skip extra entries not needed for last batch
            if file_counter >= files: break
        batch = torch.stack(batch_inp, dim=0).to(device)
        attention = model(batch)[-1]

        # for l_i, l in enumerate(attention):
        #     head_matrices = []
        #     for p_i in range(len(batch_inp)):
        #         for h_i in range(num_heads):
        #             head_matrices.append(torch.Tensor.tolist(l[p_i][h_i]))
        #         heads_dict = {
        #             "heads": head_matrices
        #         }
        #         table = pa.Table.from_pydict(heads_dict)
        #         name = f"./head_matrices/{model_shortname}_{lang}_file{file_counter}_layer{l_i}_head_matrices"
        #         pq.write_to_dataset(table, root_path=name)
        #         print(f"saved layer {l_i}, progress {file_counter}/{files} files", flush=True)

        pred_head_scores = []
        for l_i, l in enumerate(attention):
            layer_head_scores = []
            for p_i in range(len(batch_inp)):
                for h_i in range(num_heads):
                    layer_head_scores.append(head_summed_scores(torch.Tensor.tolist(l[p_i][h_i])))
            pred_head_scores.append(layer_head_scores)
            print(f"layer {l_i} processed, {file_counter}/{files} files", flush=True)
        scores_dict = {
            "head_sums": pred_head_scores
        }
        ttype = "start"
        table = pa.Table.from_pydict(scores_dict)
        name = f"./head_matrices/{ttype}/{model_shortname}_{lang}_file{start_f+file_counter}_head_summed_scores"
        pq.write_to_dataset(table, root_path=name)
        print(f"saved {file_counter}/{files} files", flush=True)