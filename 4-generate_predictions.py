import datasets
import torch
from transformers import AutoModelForCausalLM
from tuned_lens.nn.lenses import TunedLens
from huggingface_hub import login
import pyarrow.parquet as pq
import pyarrow as pa
import argparse
from datetime import datetime

start_time = datetime.utcnow()
print(start_time)
# Get the arguments that were passed
# Example run command: python generate_predictions.py --language="Java" --model="NinedayWang/PolyCoder-0.4B" --lens="AISE-TUDelft/PolyCoder-lens" --files=500 --file_start_index=500 --batch_size=4 --split_size=10 --input_size=1024 --pred_start_index=0 --device="cuda" --huggingface_token="./huggingface_token.txt" --max_runtime=25 > generate_predictions.log
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
parser.add_argument("--max_runtime", type=int, default=180, metavar="MaxRuntime",
                    help="Maximum runtime, helps this program to save in time and gracefully exit")
args = parser.parse_args()

model_shortnames = {
    "NinedayWang/PolyCoder-0.4B": "PolyCoder",
    "codeparrot/codeparrot-small-multi": "CodeParrot"
}
model_name = args.model
model_shortname = model_shortnames.get(model_name,model_name)
lens_name = args.lens
tokenizer_name = model_name
language = args.language
files = args.files
file_start_index = args.file_start_index
batch_size = args.batch_size
split_size = args.split_size
input_size = args.input_size
pred_start_index = args.pred_start_index
device_type = args.device
max_runtime = args.max_runtime

print(args)

huggingface_token_filename = args.huggingface_token
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

sample_ids = []
original_contents = []
stripped_contents  = []
tokenizeds = []
tokenized_sections = []
subsection_booleans = []
predictions = []
probabilities = []


def save_and_clear():
    global sample_ids
    global original_contents
    global stripped_contents
    global tokenizeds
    global tokenized_sections
    global subsection_booleans
    global predictions
    global probabilities
    global split_counter

    predictions_dict = {
        "id": sample_ids,
        "original_content": original_contents,
        "stripped_content": stripped_contents,
        "tokenized": tokenizeds,
        "tokenized_section": tokenized_sections,
        "subsection_boolean": subsection_booleans,
        "predictions": predictions,
        "probabilities": probabilities,
    }
    table = pa.Table.from_pydict(predictions_dict)
    name = f"./Predictions/{model_shortname}/CodeShop{language}/{model_shortname}_CodeShop{language}_l{input_size}_f{split_size}_fs{file_start_index}_ps{pred_start_index}_predictions_{split_counter}"
    pq.write_to_dataset(table, root_path=name)
    split_counter += 1
    print(f"saved split {split_counter}, progress {file_counter}/{files} files", flush=True)

    sample_ids = []
    original_contents = []
    stripped_contents  = []
    tokenizeds = []
    tokenized_sections = []
    subsection_booleans = []
    predictions = []
    probabilities = []


file_counter = 0
split_counter = 0
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
    )
    model.eval()
    device = torch.device(device_type)
    model.to(device)
    tuned_lens = TunedLens.load(lens_name)
    tuned_lens.to(device)

    dataset = datasets.load_from_disk(f"./PreProcessed/{model_shortname}/CodeShop{language}{model_shortname}PreProcessed/")

    while file_counter < files:
        time_elapsed = (datetime.utcnow()-start_time).seconds/60
        average_time = time_elapsed/(file_counter+1)
        if (time_elapsed+average_time > max_runtime):
            print(f"Time left: {max_runtime-time_elapsed}, average file runtime: {average_time} is too high to do another file, exiting.")
            save_and_clear()
            print(f"Gracefully exited due to time limit {time_elapsed}, results are saved.")
            exit()
        current_file = file_start_index+file_counter
        print(f"{time_elapsed} mins: starting file {file_counter}: sample {current_file}")
        inp = dataset[current_file]["tokenized_section"]
        file_sample_id = dataset[current_file]["id"]
        file_original_content = dataset[current_file]["original_content"]
        file_stripped_content = dataset[current_file]["stripped_content"]
        file_tokenized = dataset[current_file]["tokenized"]
        file_tokenized_section = dataset[current_file]["tokenized_section"]
        file_subsection_booleans = len(dataset[current_file]["stripped_content"]) == len(dataset[current_file]["original_content"])
        inp_length = len(inp)
        pred_index = pred_start_index
        file_predictions = None
        file_probabilities = None
        while pred_index+input_size < inp_length:
            batch_inp = []
            for p_i in range(batch_size):
                pred_tokens = torch.IntTensor(inp[pred_index:pred_index+input_size])
                batch_inp.append(pred_tokens)
                pred_index += 1
# convert batch input to correct tensor shape
batch = torch.stack(batch_inp, dim=0).to(device)
# make predictions
outputs = model(batch).hidden_states
# num_layers + 1 (embedding included)
num_outputs = len(outputs)     
if file_predictions == None:
    file_predictions = []
    for l in range(num_outputs-1):
        file_predictions.append([])
if file_probabilities == None:
    file_probabilities = []
    for l in range(num_outputs-1):
        file_probabilities.append([])
        # layer, pred, token, only last token column is relevant
        # last column = next prediction with full 1024 left context
        # skip input layer
        for l_i in range(1, num_outputs):
            layer_pred_ids = []
            layer_pred_probs = []
            for p_i in range(len(batch_inp)):
                h_state = outputs[l_i][p_i][-1]
                if l_i < num_outputs-1:
                    logit = tuned_lens.forward(h_state, l_i)
                else:
                    # final layer, have to treat differently
                    # no forward pass (no further layers), just interpret
                    logit = tuned_lens.to_logits(h_state)
                # Greedy decoding: get token with max probability
                normalized_logit = torch.exp(logit.log_softmax(dim=-1))
                pred_id = normalized_logit.argmax().item()
                pred_prob = normalized_logit[pred_id].item()
                layer_pred_ids.append(pred_id)
                layer_pred_probs.append(pred_prob)
            file_predictions[l_i-1] += layer_pred_ids
            file_probabilities[l_i-1] += layer_pred_probs

        sample_ids.append(file_sample_id)
        original_contents.append(file_original_content)
        stripped_contents.append(file_stripped_content)
        tokenizeds.append(file_tokenized)
        tokenized_sections.append(file_tokenized_section)
        subsection_booleans.append(file_subsection_booleans)
        predictions.append(file_predictions)
        probabilities.append(file_probabilities)

        file_counter += 1

        if file_counter >= files or ((split_size != None) and len(predictions) >= split_size):
            save_and_clear()
