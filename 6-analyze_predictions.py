import datasets
import torch
from transformers import AutoModelForCausalLM
from tuned_lens.nn.lenses import TunedLens
from huggingface_hub import login
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import argparse
from language_defined_tokens import (
    java_reserved_tokens,
    julia_reserved_tokens,
    go_reserved_tokens,
    kotlin_reserved_tokens,
    cpp_reserved_tokens,
    python_reserved_tokens,
    UNDEFINED_TOKEN_TYPE,
    LANGUAGE_DEFINED,
    KEYWORD_TYPE,
)

# Get the arguments that were passed
# Example run command: python analyse_predictions.py --language="Java" --model="NinedayWang/PolyCoder-0.4B" --lens="AISE-TUDelft/PolyCoder-lens" --files=500 --file_start_index=500 --batch_size=4 --split_size=10 --input_size=1024 --pred_start_index=0 --device="cuda" --huggingface_token="./huggingface_token.txt" > generate_predictions.log
parser = argparse.ArgumentParser(prog="run_dataset", add_help=True)
parser.add_argument("--language", type=str, choices=["CPP", "Go", "Java", "Julia", "Kotlin", "Python"], default="Java",
                    metavar="Lang", help="the language of the input dataset")
parser.add_argument("--model", type=str, default="NinedayWang/PolyCoder-0.4B", metavar="Mod",
                    help="the model that will be viewed with the lens")
parser.add_argument("--lens", type=str, default="AISE-TUDelft/PolyCoder-lens", metavar="Lens",
                    help="the lens that will be used to view the model's layer outputs")
parser.add_argument("--files", type=int, default=500, metavar="Files",
                    help="number of sample files to run")
parser.add_argument("--file_start_index", type=int, default=0, metavar="FileStart",
                    help="dataset sample file index to start at")
parser.add_argument("--batch_size", type=int, default=4, metavar="BatchSize",
                    help="prediction batch size, limited by GPU VRAM")
parser.add_argument("--split_size", type=int, default=10, metavar="SplitSize",
                    help="samples per output datafile, also acts like intermediate saving.")
parser.add_argument("--input_size", type=int, default=1024, metavar="InputSize",
                    help="Amount of left context for predictions.")
parser.add_argument("--pred_start_index", type=int, default=0, metavar="PredStart",
                    help="Which token predictions should start at inside files.")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", metavar="Device",
                    help="Torch device to use for calculations.")
parser.add_argument("--huggingface_token", type=str, default="./huggingface_token.txt", metavar="HFToken",
                    help="Huggingface token to load model, lens and dataset if non-local name.")
parser.add_argument("--split_counter", type=int, default=1, metavar="Splits",
                    help="How many separate dataset files the data is spread across")
args = parser.parse_args()

model_shortnames = {
    "NinedayWang/PolyCoder-0.4B": "PolyCoder",
    "codeparrot/codeparrot-small-multi": "CodeParrot"
}

model_name = args.model
model_shortname = model_shortnames.get(model_name,model_name)
lens_name = args.lens
tokenizer_name = model_name
files = args.files
file_start_index = args.file_start_index
batch_size = args.batch_size
split_size = args.split_size
input_size = args.input_size
pred_start_index = args.pred_start_index
device_type = args.device
split_counter = args.split_counter

model_name = "NinedayWang/PolyCoder-0.4B"
model_shortname = model_shortnames.get(model_name,model_name)
lens_name = "AISE-TUDelft/PolyCoder-lens"
tokenizer_name = model_name
files = args.files
input_size = 1024
split_size = 10
file_start_index = 500
pred_start_index = 0
batch_size = 4
device_type = 'cuda'
split_counter_start = 0
split_counter = 12

huggingface_token_filename = args.huggingface_token
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

languages = ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"]

data = {}
data_lengths = []
for lang in languages:
    name = f"./Predictions/{model_shortname}/CombinedPredictions{lang}"
    data[lang] = datasets.load_dataset(name, split="train")
    length = len(data[lang])
    print(f"{lang} samples: {length}", flush=True)
    data_lengths.append(length)
min_samples = np.min(data_lengths)
print(min_samples)

# for lang in languages:
#     lang_data = None
#     for split_i in range(split_counter_start, split_counter):
#         name = f"./Predictions/{model_shortname}/CodeShop{lang}/{model_shortname}_CodeShop{lang}_l{input_size}_f{split_size}_fs{file_start_index}_ps{pred_start_index}_predictions_{split_i}"
#         # Load full dataset into memory
#         dataset_init = datasets.load_dataset(name, split="train")
#         if lang_data == None:
#             lang_data = dataset_init
#         else:
#             lang_data = datasets.concatenate_datasets([lang_data, dataset_init])
#     data[lang] = lang_data

    # print(f"{lang} samples: {len(data[lang])}", flush=True)

reserved_tokens = {
    "CPP": cpp_reserved_tokens,
    "Go": go_reserved_tokens,
    "Java": java_reserved_tokens,
    "Julia": julia_reserved_tokens,
    "Kotlin": kotlin_reserved_tokens,
    "Python": python_reserved_tokens,
}

        # "id": sample_ids,
        # "original_content": original_contents,
        # "stripped_content": stripped_contents,
        # "tokenized": tokenizeds,
        # "tokenized_section": tokenized_sections,
        # "subsection_boolean": subsection_booleans,
        # "predictions": predictions,
        # "probabilities": probabilities,
def find_token_types(inputs, language):
    """
    None if undefined 
    """
    sample_token_types = []
    # skip first input to match up indices with predictions
    for input in inputs:
        token_type = None
        # we assume these each have their own vocabulary entry
        token_type = reserved_tokens[language].get(input, None)
        # try again with removed whitespace
        if token_type is None:
            s_input = input.replace("Ġ","")
            s_input = s_input.replace("Ċ","")
            s_input = s_input.replace("ĉ","")
            token_type = reserved_tokens[language].get(s_input, None)
        sample_token_types.append(token_type)
    return sample_token_types


lang_boxplot = []
user_boxplot = []
keyw_boxplot = []
l_correct_bars = []
u_correct_bars = []
for lang in languages:
    language_data = []
    # token_types[sample_i][token_i] = one of TOKEN_TYPE integers
    token_types = []
    # correct_predictions[sample_i][token_i][layer_i] = 1 or 0
    correct_predictions = []
    # correct_predictions[sample_i][token_i] = between 1 and num_layers
    first_correct_predictions = []
    counter = 0
    # for sample in data[lang]:
    for sample_i in range(min_samples):
        sample = data[lang][sample_i]
        sample_tokenized_section = sample["tokenized_section"]
        sample_id = sample["id"]
        input_ids = sample_tokenized_section[input_size:len(sample_tokenized_section)]
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        token_types.append(find_token_types(input_tokens, lang))
        # for checking token types
        # for i in range(len(input_tokens)):
        #     print(f"{input_tokens[i]}\t{token_types[0][i]}")
        # exit()

        predictions = sample["predictions"]

        # check correct tokens in a layer
        # for token_i in range(len(predictions[23])):
        #     for layer_i in range(len(predictions)):
        #         print(f"{input_ids[token_i]}\t{predictions[layer_i][token_i]}")
        #     break
        # break

        sample_correct_tokens = []
        sample_first_correct_tokens = []
        num_layers = len(predictions)
        num_tokens = len(predictions[0])
        # loop tokens, cant check if the last one is correct
        for token_i in range(num_tokens-1):
            # 
            first_correct = 0
            layer_correct_tokens = []
            for layer_i in range(num_layers):
                if predictions[layer_i][token_i] == input_ids[token_i]:
                    layer_correct_tokens.append(1)
                    # record first correct layer
                    if first_correct == 0: first_correct = layer_i+1
                else:
                    layer_correct_tokens.append(0)
                    # reset first correct layer if prediction changed
                    if first_correct != 0: first_correct = 0
            sample_correct_tokens.append(layer_correct_tokens)
            sample_first_correct_tokens.append(first_correct)
        correct_predictions.append(sample_correct_tokens)
        first_correct_predictions.append(sample_first_correct_tokens)
    
    language_data_dict = {
        "id": data[lang]["id"],
        "original_content": data[lang]["original_content"],
        "stripped_content": data[lang]["stripped_content"],
        "tokenized": data[lang]["tokenized"],
        "tokenized_section": data[lang]["tokenized_section"],
        "subsection_boolean": data[lang]["subsection_boolean"],
        "predictions": data[lang]["predictions"],
        "probabilities": data[lang]["probabilities"],
        "token_types": token_types,
        "correct_predictions":  correct_predictions,
        "first_correct_predictions": first_correct_predictions
    }
    table = pa.Table.from_pydict(language_data_dict)
    name = f"./Predictions/{model_shortname}/{model_shortname}CodeShop{lang}Predictions/"
    pq.write_to_dataset(table, root_path=name)
        

    # print("prediction entries: ", len(correct_predictions))
    # print("token types entries: ", len(token_types))
    # print("first correct token types entries: ", len(first_correct_predictions))
    # print("1st sample token predictions: ", len(correct_predictions[0]))
    # print("1st sample token types: ", len(token_types[0]))
    # print("1st sample first correct layers", len(first_correct_predictions[0]))

    # count total predictions
    token_count = 0
    prediction_count = 0
    for sample_i in range(len(correct_predictions)):
        for token_i in range(len(correct_predictions[sample_i])):
            token_count += 1
            for layer_i in range(len(correct_predictions[sample_i][token_i])):
                prediction_count += 1
    # print(token_count)
    # print(prediction_count)

    # count how many samples are available at each token index
    # only relevant if samples have different lengths
    sample_lengths = []
    for sample in correct_predictions:
        for token_i in range(len(sample)):
            if token_i >= len(sample_lengths):
                sample_lengths.append(1)
            else:
                sample_lengths[token_i] += 1
    # print("sample lengths: ", sample_lengths)

    # count how many samples are in each token type category
    l_sample_lengths = []
    u_sample_lengths = []
    kw_sample_lengths = []
    for sample_i in range(len(correct_predictions)):
        for token_i in range(len(correct_predictions[sample_i])):
            if token_i >= len(l_sample_lengths):
                    l_sample_lengths.append(0)
                    u_sample_lengths.append(0)
                    kw_sample_lengths.append(0)

            token_type = token_types[sample_i][token_i]
            if UNDEFINED_TOKEN_TYPE.get(token_type, False): continue
            if LANGUAGE_DEFINED.get(token_type, False):
                l_sample_lengths[token_i] += 1
                if token_type == KEYWORD_TYPE:
                    kw_sample_lengths[token_i] += 1
            else:
                # user token
                u_sample_lengths[token_i] += 1
    # print(len(l_sample_lengths))
    # print(len(u_sample_lengths))
    # print("count of language tokens ", l_sample_lengths, flush=True)
    # print("count of user tokens ", u_sample_lengths, flush=True)

    # count how many overall predictions are correct
    # correct_predictions_count[token_i][layer_i] = 0 - n_samples
    correct_predictions_count = []
    for sample in correct_predictions:
        for token_i in range(len(sample)):
            if token_i >= len(correct_predictions_count):
                blank = []
                for _ in range(num_layers):
                    blank.append(0)
                correct_predictions_count.append(blank)
            for layer_i in range(num_layers):
                correct_predictions_count[token_i][layer_i] += sample[token_i][layer_i]
    # print("count overall correct predictions: ", correct_predictions_count)

    # count how many predictions are correct in each token type category
    # x_correct_predictions_count[token_i][layer_i] = 0 - n_samples of token
    l_correct_predictions_count = []
    u_correct_predictions_count = []
    blank_layers = []
    for _ in range(num_layers):
        blank_layers.append(0)
    for sample_i in range(len(correct_predictions)):
        for token_i in range(len(correct_predictions[sample_i])):
            if token_i >= len(l_correct_predictions_count):
                l_correct_predictions_count.append(blank_layers.copy())
                u_correct_predictions_count.append(blank_layers.copy())

            token_type = token_types[sample_i][token_i]
            if UNDEFINED_TOKEN_TYPE.get(token_type, False): continue
            if LANGUAGE_DEFINED.get(token_type, False):
                # language token
                count_to_update = l_correct_predictions_count
            else:
                # user token
                count_to_update = u_correct_predictions_count
            
            for layer_i in range(num_layers):
                prediction = correct_predictions[sample_i][token_i][layer_i]
                count_to_update[token_i][layer_i] += prediction
    # show how many are correct in each layer at a specific token index
    # print("language total correct predictions: ", l_correct_predictions_count[998], flush=True)
    # print("user total correct predictions: ", u_correct_predictions_count[998], flush=True)

    # correct_fractions[token_i][layer_i] = 0 to 1
    correct_fractions = []
    for token_i in range(len(correct_predictions_count)):
        token_correct_fractions = []
        for layer_i in range(num_layers):
            total = sample_lengths[token_i]
            # print(total)
            correct = correct_predictions_count[token_i][layer_i]
            # print(correct) 
            token_correct_fractions.append(correct / total)
        correct_fractions.append(token_correct_fractions)
    # 1000 preds, cant check last one = 999, 0 indexed -> 998 to check last
    # print("overall correct fraction: ", correct_fractions[998])

    l_correct_fractions = []
    u_correct_fractions = []
    for token_i in range(len(l_correct_predictions_count)):
        l_token_correct_fractions = []
        u_token_correct_fractions = []
        for layer_i in range(num_layers):
            l_total = l_sample_lengths[token_i]
            u_total = u_sample_lengths[token_i]
            l_correct = l_correct_predictions_count[token_i][layer_i]
            u_correct = u_correct_predictions_count[token_i][layer_i]
            if l_total == 0:
                l_token_correct_fractions.append(0)
            else:
                l_token_correct_fractions.append(l_correct / l_total)
            if u_total == 0:
                u_token_correct_fractions.append(0)
            else:
                u_token_correct_fractions.append(u_correct / u_total)
        l_correct_fractions.append(l_token_correct_fractions)
        u_correct_fractions.append(u_token_correct_fractions)
    # print("language correct fraction: ", l_correct_fractions[998])
    # print("user correct fraction: ", u_correct_fractions[998])

    # missed predictions, average correct % across all tokens in final layer
    l_fraction_sum = 0
    u_fraction_sum = 0
    for token_i in range(len(l_correct_fractions)):
        l_fraction_sum += l_correct_fractions[token_i][23]
        u_fraction_sum += u_correct_fractions[token_i][23]
    l_fraction_avg = l_fraction_sum / len(l_correct_fractions)
    u_fraction_avg = u_fraction_sum / len(u_correct_fractions)
    l_correct_bars.append(l_fraction_avg)
    u_correct_bars.append(u_fraction_avg)

    # average of first correct
    average_first_correct = []
    for sample_i in range(len(first_correct_predictions)):
        for token_i in range(len(first_correct_predictions[sample_i])):
            if token_i >= len(average_first_correct):
                average_first_correct.append(0)
            first_correct_layer = first_correct_predictions[sample_i][token_i]
            average_first_correct[token_i] += first_correct_layer
    for token_i in range(len(average_first_correct)):
        total = sample_lengths[token_i]
        if total == 0:
            average_first_correct[token_i] = 0
        else:
            average = average_first_correct[token_i] / total
            average_first_correct[token_i] = average
    # print("overall average of first correct: ", average_first_correct)

    # average per token type
    l_average_first_correct = []
    u_average_first_correct = []
    kw_average_first_correct = []
    for sample_i in range(len(first_correct_predictions)):
        for token_i in range(len(first_correct_predictions[sample_i])):
            if token_i >= len(l_average_first_correct):
                l_average_first_correct.append(0)
                u_average_first_correct.append(0)
                kw_average_first_correct.append(0)
            token_type = token_types[sample_i][token_i]
            if UNDEFINED_TOKEN_TYPE.get(token_type, False): continue
            first_correct_layer = first_correct_predictions[sample_i][token_i]
            if LANGUAGE_DEFINED.get(token_type, False):
                # language token
                l_average_first_correct[token_i] += first_correct_layer
                if token_type == KEYWORD_TYPE:
                    kw_average_first_correct[token_i] += first_correct_layer
            else:
                # user token
                u_average_first_correct[token_i] += first_correct_layer
    
    l_token_total = 0
    u_token_total = 0
    kw_token_total = 0
    for sample_i in range(len(first_correct_predictions)):
        for token_i in range(len(first_correct_predictions[sample_i])):
            token_type = token_types[sample_i][token_i]
            if UNDEFINED_TOKEN_TYPE.get(token_type, False): continue
            if LANGUAGE_DEFINED.get(token_type, False):
                # language token
                l_token_total += 1
                if token_type == KEYWORD_TYPE:
                    kw_token_total += 1
            else:
                # user token
                u_token_total += 1
    print(f"{lang} total lang: {l_token_total}, total kw: {kw_token_total}, total user: {u_token_total}")

    for token_i in range(len(l_average_first_correct)):
        l_total = l_sample_lengths[token_i]
        if l_total == 0:
            l_average_first_correct[token_i] = 0
        else:
            average = l_average_first_correct[token_i] / l_total
            l_average_first_correct[token_i] = average
    for token_i in range(len(u_average_first_correct)):
        u_total = u_sample_lengths[token_i]
        if u_total == 0:
            u_average_first_correct[token_i] = 0
        else:
            average = u_average_first_correct[token_i] / u_total
            u_average_first_correct[token_i] = average
    # print("language average of first correct: ", l_average_first_correct)
    # print("user average of first correct: ", u_average_first_correct)
    for token_i in range(len(kw_average_first_correct)):
        kw_total = kw_sample_lengths[token_i]
        if kw_total == 0:
            kw_average_first_correct[token_i] = 0
        else:
            average = kw_average_first_correct[token_i] / kw_total
            kw_average_first_correct[token_i] = average
    
    x = []
    y = []
    for token_i in range(len(u_average_first_correct)):
    # for token_i in range(500,2500):
        if u_average_first_correct[token_i] != 0:
            x.append(u_average_first_correct[token_i])
            y.append(token_i)
    user_boxplot.append(x)
    plt.figure(figsize=(20,5))
    # plt.xlabel('Amount of Tokens of Context Available')
    # plt.ylabel('Average First Correct Layer')
    plt.ylim([0,15])
    plt.plot(y,x, "ro")
    plt.savefig(f"{lang}_user_defined_elements.png")
    plt.close()
    x = []
    y = []
    for token_i in range(len(l_average_first_correct)):
    # for token_i in range(500,2500):
        if l_average_first_correct[token_i] != 0:
            x.append(l_average_first_correct[token_i])
            y.append(token_i)
    lang_boxplot.append(x)
    plt.figure(figsize=(20,5))
    # plt.xlabel('Amount of Tokens of Context Available')
    # plt.ylabel('Average First Correct Layer')
    # txt="Average First Correct layer Depth for Language Defined Keywords in Java Samples, "
    # plt.figtext(0.5, 0.00, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.ylim([0,15])
    plt.plot(y,x, "go")
    plt.savefig(f"{lang}_lang_defined_elements.png")
    plt.close()
    x = []
    y = []
    for token_i in range(len(kw_average_first_correct)):
    # for token_i in range(500,2500):
        if kw_average_first_correct[token_i] != 0:
            x.append(kw_average_first_correct[token_i])
            y.append(token_i)
    keyw_boxplot.append(x)
    plt.figure(figsize=(20,5))
    # plt.xlabel('Amount of Tokens of Context Available')
    # plt.ylabel('Average First Correct Layer')
    # txt="Average First Correct layer Depth for Language Defined Keywords in Java Samples, "
    # plt.figtext(0.5, 0.00, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.ylim([0,15])
    plt.plot(y,x, "go")
    plt.savefig(f"{lang}_keyw_elements.png")
    plt.close()
    
all_boxplots = []
colors = []
pos1 = []
pos2 = []
pos3 = []
widths = []
labels = []
barWidth = 0.25
for i in range(len(lang_boxplot)):
    widths.append(barWidth)
    widths.append(barWidth)
    widths.append(barWidth)
    all_boxplots.append(lang_boxplot[i])
    colors.append('#00FF00')
    pos1.append(i*1.1)
    all_boxplots.append(keyw_boxplot[i])
    colors.append('#99CCFF')
    pos2.append(i*1.1+barWidth)
    all_boxplots.append(user_boxplot[i])
    colors.append('#FFFF00')
    pos3.append(i*1.1+2*barWidth)

fig = plt.figure(figsize =(14, 7))

ax = fig.add_subplot(111)
bp1 = ax.boxplot(lang_boxplot, positions=pos1, notch=True, widths=barWidth, 
                 patch_artist=True, boxprops=dict(facecolor="#00FF00"))
bp2 = ax.boxplot(keyw_boxplot, positions=pos2, notch=True, widths=barWidth, 
                 patch_artist=True, boxprops=dict(facecolor="#99CCFF"))
bp3 = ax.boxplot(user_boxplot, positions=pos3, notch=True, widths=barWidth, 
                 patch_artist=True, boxprops=dict(facecolor="#FFFF00"))
for median in bp1['medians']:
    median.set_color('black')
for median in bp2['medians']:
    median.set_color('black')
for median in bp3['medians']:
    median.set_color('black')
# ax.legend(ax.get_legend_handles_labels(), ['Language Defined', 'Strict Keyword', 'User Defined'], loc='upper right')

# fig = plt.figure(figsize =(15, 7))
# ax = fig.add_subplot(111)
# # Creating plot
# bp = ax.boxplot(all_boxplots, patch_artist = True, 
#     positions=positions, widths=widths,
#     notch=True, vert = 1
# )


# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)

plt.ylim([0, 12])
plt.xticks([r*1.1 + barWidth for r in range(len(languages))],
        ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"])
# title = plt.title("Average First Correct Layer per Language, lower is better.\n Language Defined (green), Strict Keyword (blue) and User Defined (yellow) Elements.\n Arranged from most trained on to least trained on, Julia and Kotlin were not trained on at all.")
# plt.setp(title, fontsize=18)
# plt.legend()
colors = {'Language Defined':'#00FF00', 'Strict Keyword':'#99CCFF', 'User Defined':'#FFFF00'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.xlabel('Language and Token Type', fontweight ='bold', fontsize = 14)
plt.ylabel('Average Layer Depth of First Correct Prediction', fontweight ='bold', fontsize = 14)
plt.legend(handles, labels, fontsize=13, loc="upper right")
plt.savefig(f"Element_Types_BP.png")

fig = plt.figure(figsize =(15, 7))
ax = fig.add_subplot(111)
# Creating plot
bp = ax.boxplot(user_boxplot, patch_artist = True,
                notch ='True', vert = 1)

colors = ['#99CCFF', '#99CCFF', '#99CCFF', '#99CCFF', '#FFB366', '#FFB366']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylim([0, 12])
xtickNames = plt.setp(ax, xticklabels=languages)
plt.setp(xtickNames, fontsize=16)
title = plt.title("Average First Correct Layer per language for User Defined Elements, lower is better.\n Arranged from most trained on to least trained on, orange Julia and Kotlin were not trained on at all.")
plt.setp(title, fontsize=18)
plt.savefig(f"User_defined_elements_bp.png")


fig = plt.figure(figsize =(15, 7))
ax = fig.add_subplot(111)
# Creating plot
bp = ax.boxplot(lang_boxplot, patch_artist = True,
                notch ='True', vert = 1)

colors = ['#99CCFF', '#99CCFF', '#99CCFF', '#99CCFF', '#FFB366', '#FFB366']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylim([0, 12])
xtickNames = plt.setp(ax, xticklabels=languages)
plt.setp(xtickNames, fontsize=16)
title = plt.title("Average First Correct Layer per language for Language Defined Elements, lower is better.\n Arranged from most trained on to least trained on, orange Julia and Kotlin were not trained on at all.")
plt.setp(title, fontsize=18)
plt.savefig(f"Language_defined_elements_bp.png")

fig = plt.figure(figsize =(15, 7))
ax = fig.add_subplot(111)
# Creating plot
bp = ax.boxplot(keyw_boxplot, patch_artist = True,
                notch ='True', vert = 1)

colors = ['#99CCFF', '#99CCFF', '#99CCFF', '#99CCFF', '#FFB366', '#FFB366']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.ylim([0, 12])
xtickNames = plt.setp(ax, xticklabels=languages)
plt.setp(xtickNames, fontsize=16)
title = plt.title("Average First Correct Layer per language for Language Defined Elements, lower is better.\n Arranged from most trained on to least trained on, orange Julia and Kotlin were not trained on at all.")
plt.setp(title, fontsize=18)
plt.savefig(f"Strict_keyword_elements_bp.png")


prediction_stats = {
    "user_boxplot_data": user_boxplot,
    "lang_boxplot_data": lang_boxplot,
}
table = pa.Table.from_pydict(prediction_stats)
name = f"./PredictionStats"
pq.write_to_dataset(table, root_path=name)


# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(15, 7))
 
# Set position of bar on X axis
br1 = np.arange(len(l_correct_bars))
br2 = [x + barWidth for x in br1]

plt.ylim([0, 1])
# Make the plot
plt.bar(br1, l_correct_bars, color ='#00FF00', width = barWidth,
        edgecolor ='grey', label ='Language Elements')
plt.bar(br2, u_correct_bars, color ='#FFFF00', width = barWidth,
        edgecolor ='grey', label ='User Elements')
 
# Adding Xticks
plt.xlabel('Language', fontweight ='bold', fontsize = 15)
plt.ylabel('Fraction Correct', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(l_correct_bars))],
        ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"])
title = plt.title("Fraction of correct Final Predictions per Language and Token Element Type. \nArranged from most trained on to least trained on, yellow Julia and Kotlin were not trained on at all.")
plt.setp(title, fontsize=18)
 
plt.legend()
plt.savefig(f"Fraction_Final_Correct.png")