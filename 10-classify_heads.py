# defines a method to classify null heads
# using a pre-calculated distribution of summed attention scores
# a head is a null head if:
#     the first token has unusual attention: in some top percentile
#     all other tokens do not have unusual attention: in some bottom percentile
# we need a pre-calculated expected summed score distribution because:
#   - large input sizes (1024x1024) are not observable for validation
#   - the expected summed score distribution changes with input size
#   - the expected summed score distribution varies between token index
# this means for other tokens we cannot pick a single cut off value, we need one for each token index, matching the expected distribution.

import datasets
import numpy as np
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

model_name = 'NinedayWang/PolyCoder-0.4B'
model_shortname = 'polycoder'
tokenizer_name = model_name
input_size = 1024
num_heads = 16
num_layers = 24


def get_perct(scores, perct):
    n = len(scores)
    index = perct*(n-1)
    q, mod = divmod(index, 1)
    if mod == 0:
        return scores[int(q)]
    else:
        return (scores[int(q)]+mod*(scores[int(q+1)]-scores[int(q)]))
def get_mean(scores):
    sum = 0
    for s in scores:
        sum += s
    return sum / len(scores)


head_distribution = datasets.load_dataset(f"./HeadDistribution/{input_size}", split="train")["head_distribution"]

x = [20, 94, 248, 335, 507, 818, 937, 963, 989, 1008, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]
y = [119.5931372, 98.37370223, 81.79457761, 79.54779794, 66.99215217, 53.41278356, 26.5400807, 23.91724212, 14.12568977, 8.985731186, 7.389622167, 7.094203979, 6.601636016, 4.787592463, 4.601742387, 3.979743347, 3.259321902, 2.609477952, 2.008372158, 1.237564623, 0.749806643]

# approximation function of highest observed attention per token
approx = [1024]
for t_i in range(1,20):
    approx.append(1024/(2*math.log(t_i+1,2)))
for x_i in range(len(x)-1):
    count = x[x_i+1] - x[x_i]
    diff = y[x_i+1] - y[x_i]
    incr = diff / count
    current = y[x_i]
    for t_i in range(count):
        approx.append(current)
        current += incr
approx.append(y[-1])


num_samples = 60
lang = "Go"

else_files = []
start_files = []
for f_i in range(num_samples):
    name = f"./head_matrices/start/{model_shortname}_{lang}_file{f_i+1}_head_summed_scores"
    heads = datasets.load_dataset(name, split="train")["head_sums"]
    start_files.append(heads)
    name = f"./head_matrices/else/{model_shortname}_{lang}_file{f_i+1}_head_summed_scores"
    heads = datasets.load_dataset(name, split="train")["head_sums"]
    else_files.append(heads)

first_token_sums = []
for f_i in range(num_samples):
    heads = else_files[f_i]
    head_res = []
    for layer in heads:
        layer_res = []
        for head in layer:
            first_token_sums.append(head[0])
    heads = start_files[f_i]
    head_res = []
    for layer in heads:
        layer_res = []
        for head in layer:
            first_token_sums.append(head[0])
first_token_sums = np.asarray(first_token_sums)
first_token_sums.sort()

print("count: ", len(first_token_sums))
print("mean: ", first_token_sums.mean())
print("min: ", first_token_sums[0])
print("1st: ", get_perct(first_token_sums, 0.25))
print("median: ", get_perct(first_token_sums, 0.5))
print("3rd: ", get_perct(first_token_sums, 0.75))
print("max: ", first_token_sums[-1])

def classify_null_head(head_sums):
    # check if first is in top 50% of attention on first
    first_t_min = get_perct(first_token_sums, 0.95)
    first_t_check = head_sums[0] > first_t_min
    if not first_t_check:
        return 0
    # check if all others are below a chosen fraction of max observed attention
    # for token_i in range(1,len(head_sums)):
    #     score = head_sums[token_i]
    #     max_baseline = 0.5
    #     frac_of_max_observed = 0.25
    #     max_score = max_baseline+frac_of_max_observed*approx[token_i]
    #     if score > max_score:
    #         return 0
    return 1

else_null_heads = np.zeros((24,16))
for f_i in range(num_samples):
    heads = else_files[f_i]
    head_res = []
    for l_i, layer in enumerate(heads):
        layer_res = []
        for h_i, head in enumerate(layer):
            head_type = classify_null_head(head)
            layer_res.append(head_type)
            else_null_heads[l_i,h_i] += head_type
        head_res.append(layer_res)
print("-------")
else_null_heads_layer = np.zeros(24)
for l_i, l in enumerate(else_null_heads):
    print(l)
    for v in l:
        else_null_heads_layer[l_i] += v
print(else_null_heads_layer)

start_null_heads = np.zeros((24,16))
for f_i in range(num_samples):
    heads = start_files[f_i]
    head_res = []
    for l_i, layer in enumerate(heads):
        layer_res = []
        for h_i, head in enumerate(layer):
            head_type = classify_null_head(head)
            layer_res.append(head_type)
            start_null_heads[l_i,h_i] += head_type
        head_res.append(layer_res)
print("-------")

start_null_heads_layer = np.zeros(24)
for l_i, l in enumerate(start_null_heads):
    print(l)
    for v in l:
        start_null_heads_layer[l_i] += v
print(start_null_heads_layer)

# set width of bar
barWidth = 0.1
fig = plt.subplots(figsize =(15, 7))
 
# Set position of bar on X axis
br1 = np.arange(24)*0.5
br2 = [x + barWidth for x in br1]

plt.ylim([0, 0.25])
# Make the plot
plt.bar(br1, else_null_heads_layer/(num_heads*num_samples), color ='#00FF00', width = barWidth,
        edgecolor ='grey', label ='Language Elements')
plt.bar(br2, start_null_heads_layer/(num_heads*num_samples), color ='#FFFF00', width = barWidth,
        edgecolor ='grey', label ='User Elements')
 
# Adding Xticks
plt.xlabel('Model Layer', fontweight ='bold', fontsize = 15)
plt.ylabel('Fraction Of Null Heads', fontweight ='bold', fontsize = 15)
plt.xticks([0.5*r + barWidth for r in range(24)],
        range(1,25))
title = plt.title("Fraction of Null Attention Heads per Layer for 'else' and 'start' Tokens")
plt.setp(title, fontsize=18)
 
plt.legend()
plt.savefig(f"{lang}_Null_Attention_Heads.png")