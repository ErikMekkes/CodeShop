import datasets
from matplotlib import pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub import login

huggingface_token_filename = "huggingface_token.txt"
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)
tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-0.4B")


model_shortname = "PolyCoder"
languages = ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"]
# lang tokens and their variant with a space character in front
return_token_ids = [397, 429]
continue_token_ids = [3691, 3888]
else_token_ids =  [599, 730]
for_token_ids =  [529, 446]
if_token_ids = [285, 392]
assign_token_ids = [31, 260, 1006, 2611, 2846, 3015, 5210, 10992]
# user tokens and their variant with a space character in front
# also including the capitalized first letter variant with and without space
key_token_ids = [689,1255,1031,4345]
keys_token_ids = [4067,5232,5188,25131]
value_token_ids = [731,734,842,3320] # value
sum_token_ids = [1157,4962,5463,16078]
start_token_ids = [1182,1589,2163,5434,4172]
message_token_ids = [2182,2057,1564,5223]

        # "id": data[lang]["id"],
        # "original_content": data[lang]["original_content"],
        # "stripped_content": data[lang]["stripped_content"],
        # "tokenized": data[lang]["tokenized"],
        # "tokenized_section": data[lang]["tokenized_section"],
        # "subsection_boolean": data[lang]["subsection_boolean"],
        # "predictions": data[lang]["predictions"],
        # "probabilities": data[lang]["probabilities"],
        # "token_types": token_types,
        # "correct_predictions":  correct_predictions,
        # "first_correct_predictions": first_correct_predictions

# lang tokens and their variant with a space character in front

input_size = 1024

data = {}
interest = {}
for lang in languages:
    continue_tokens = [[],[],[],[]]
    return_tokens = [[],[],[],[]]
    else_tokens = [[],[],[],[]]
    for_tokens = [[],[],[],[]]
    if_tokens = [[],[],[],[]]
    assign_tokens = [[],[],[],[]]
    key_tokens = [[],[],[],[]]
    value_tokens = [[],[],[],[]]
    sum_tokens = [[],[],[],[]]
    start_tokens = [[],[],[],[]]
    message_tokens = [[],[],[],[]]
    name = f"./Predictions/{model_shortname}/{model_shortname}CodeShop{lang}Predictions/"
    data[lang] = datasets.load_dataset(name, split="train")
    print(f"{lang} samples: {len(data[lang])}", flush=True)
    
    for sample_i in range(200):
        sample = data[lang][sample_i]
        sample_id = sample["id"]
        for token_i in range(input_size,len(sample["tokenized_section"])-1):
            token = sample["tokenized_section"][token_i]
            # token_str = tokenizer.convert_ids_to_tokens([token])[0]
            # previous_token_id = sample["tokenized_section"][token_i-1]
            # previous_token = tokenizer.convert_ids_to_tokens([previous_token_id])[0]
            if token_i < len(sample["tokenized_section"])-2:
                next_id = sample["tokenized_section"][token_i+1]
                next_token = tokenizer.convert_ids_to_tokens([next_id])[0]
            first_cor = sample["first_correct_predictions"][token_i-input_size]
            if first_cor == 0: continue
            to_add_to = None
            if token in continue_token_ids:
                # print("return", token_i, token, sample["first_correct_predictions"][token_i-input_size], sample["predictions"][23][token_i-input_size])
                to_add_to = continue_tokens
            if token in return_token_ids:
                to_add_to = return_tokens
            if token in else_token_ids:
                to_add_to = else_tokens
            if token in for_token_ids:
                # not one of these = split up word
                if next_token[0] == 'Ġ' or next_token[0] == '(':
                    to_add_to = for_tokens
            if token in if_token_ids:
                to_add_to = if_tokens
                # print(previous_token, token_str, next_token, first_cor)
            if token in assign_token_ids:
                to_add_to = assign_tokens
            if token in key_token_ids:
                to_add_to = key_tokens
            if token in value_token_ids:
                to_add_to = value_tokens
            if token in sum_token_ids:
                to_add_to = sum_tokens
            if token in start_token_ids:
                to_add_to = start_tokens
            if token in message_token_ids:
                to_add_to = message_tokens
            if to_add_to != None:
                to_add_to[0].append(sample_id)
                to_add_to[1].append(token_i)
                to_add_to[2].append(first_cor)
                to_add_to[3].append(sample_i)
    
    interest[lang] = [
        # continue_tokens,
        return_tokens,
        else_tokens,
        # for_tokens,
        if_tokens,
        assign_tokens,
        key_tokens,
        value_tokens,
        # sum_tokens,
        start_tokens,
        message_tokens,
    ]
    # interest[lang] = {
    #     "return_tokens" : return_tokens,
    #     "else_tokens" : else_tokens,
    #     "for_tokens" : for_tokens,
    #     "if_tokens" : if_tokens,
    #     "assign_tokens" : assign_tokens,
    #     "key_tokens" : key_tokens,
    #     "value_tokens" : value_tokens,
    #     "sum_tokens" : sum_tokens,
    #     "start_tokens" : start_tokens,
    #     "message_tokens" : message_tokens,
    # }

    table = pa.Table.from_pydict(interest)
    name = f"./Interest/{lang}TokensOfInterest"
    pq.write_to_dataset(table, root_path=name)

    print(len(continue_tokens[2]))
    print(len(return_tokens[2]))
    print(len(else_tokens[2]))
    print("for: ", len(for_tokens[2]))
    print("if: ", len(if_tokens[2]))
    print(len(key_tokens[2]))
    print(len(value_tokens[2]))
    print(len(sum_tokens[2]))
    print(len(start_tokens[2]))
    print(len(message_tokens[2]))
    res = [
        # continue_tokens[2],
        return_tokens[2],
        else_tokens[2],
        # for_tokens[2],
        if_tokens[2],
        assign_tokens[2],
        key_tokens[2],
        value_tokens[2],
        # sum_tokens[2],
        start_tokens[2],
        message_tokens[2]
    ]
    fig = plt.figure(figsize =(20, 12))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(res, patch_artist = True,
                    notch = False, vert = 0,
                    # widths=0.25
    )
    for median in bp['medians']:
        median.set_color('black')

    colors = ['#00FF00', '#00FF00', '#00FF00', '#00FF00', '#FFFF00','#FFFF00', '#FFFF00','#FFFF00']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlim([0, 24])
    
    token_labels = [
        # f"continue n={len(continue_tokens[2])}",
        f"'return' n={len(return_tokens[2])}",
        f"'else' n={len(else_tokens[2])}",
        # f"for n={len(for_tokens[2])}",
        f"'if' n={len(if_tokens[2])}",
        f"'=' n={len(assign_tokens[2])}",
        f"'key' n={len(key_tokens[2])}",
        f"'value' n={len(value_tokens[2])}",
        # f"sum n={len(sum_tokens[2])}",
        f"'start' n={len(start_tokens[2])}",
        f"'message' n={len(message_tokens[2])}"
    ]
    # ytickNames = plt.setp(ax, yticklabels=range(0,12))
    plt.setp(ax.get_xticklabels(), fontsize=22)
    ytickNames = plt.setp(ax, yticklabels=token_labels)
    plt.setp(ytickNames, fontsize=22, rotation=0, ha="right")
    plt.tight_layout()
    
    colors = {'Language':'#00FF00', 'User':'#FFFF00'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.gcf().subplots_adjust(bottom=0.08)
    plt.legend(handles, labels, fontsize=22, loc="upper right")
    plt.xlabel('Average Layer Depth of First Correct Prediction', fontweight ='bold', fontsize = 22)
    # title = plt.title("Average First Correct Layer per token of interest, lower is better.\n Green Language defined elements on the left, Yellow User defined elements on the right.")
    # plt.setp(title, fontsize=18)
    plt.savefig(f"./Interest/{lang}_tokens_of_interest.png")

