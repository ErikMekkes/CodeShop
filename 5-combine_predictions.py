import datasets
import argparse
import pyarrow.parquet as pq
import pyarrow as pa

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



# in order of most trained on
languages = ["Java", "CPP", "Python", "Go", "Julia", "Kotlin"]
# languages = ["CPP"]
files = {
    "Python": [
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_0",
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_1",
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_2",
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_3",
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_4",
        "PolyCoder_CodeShopPython_l1024_f10_fs500_ps0_predictions_5",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_0",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_1",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_2",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_3",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_4",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_5",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_6",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_7",
        "PolyCoder_CodeShopPython_l1024_f10_fs560_ps0_predictions_8",
        "PolyCoder_CodeShopPython_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopPython_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopPython_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopPython_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopPython_l1024_f20_fs0_ps0_predictions_4",
    ],
    "Kotlin": [
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_0",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_2",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_3",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_4",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_5",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_6",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_7",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_8",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_9",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_10",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_11",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_12",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_13",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_14",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_15",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_16",
        "PolyCoder_CodeShopKotlin_l1024_f10_fs500_ps0_predictions_17",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_4",
        "PolyCoder_CodeShopKotlin_l1024_f20_fs0_ps0_predictions_5",
    ],
    "Java": [
        "PolyCoder_CodeShopJava_l1024_f10_fs500_ps0_predictions_0",
        "PolyCoder_CodeShopJava_l1024_f10_fs500_ps0_predictions_1",
        "PolyCoder_CodeShopJava_l1024_f10_fs500_ps0_predictions_2",
        "PolyCoder_CodeShopJava_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopJava_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopJava_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopJava_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopJava_l1024_f20_fs0_ps0_predictions_4",
        "PolyCoder_CodeShopJava_l1024_f20_fs100_ps0_predictions_0",
        "PolyCoder_CodeShopJava_l1024_f20_fs100_ps0_predictions_1",
        "PolyCoder_CodeShopJava_l1024_f20_fs100_ps0_predictions_2",
        "PolyCoder_CodeShopJava_l1024_f20_fs100_ps0_predictions_3",
        "PolyCoder_CodeShopJava_l1024_f20_fs100_ps0_predictions_4",
    ],
    "Go": [
        "PolyCoder_CodeShopGo_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopGo_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopGo_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopGo_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopGo_l1024_f20_fs0_ps0_predictions_4",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_0",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_1",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_2",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_3",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_4",
        "PolyCoder_CodeShopGo_l1024_f20_fs100_ps0_predictions_5",
    ],
    "CPP": [
        "PolyCoder_CodeShopCPP_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopCPP_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopCPP_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopCPP_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopCPP_l1024_f20_fs0_ps0_predictions_4",
        "PolyCoder_CodeShopCPP_l1024_f20_fs100_ps0_predictions_0",
        "PolyCoder_CodeShopCPP_l1024_f20_fs100_ps0_predictions_1",
        "PolyCoder_CodeShopCPP_l1024_f20_fs100_ps0_predictions_2",
        "PolyCoder_CodeShopCPP_l1024_f20_fs100_ps0_predictions_3",
        "PolyCoder_CodeShopCPP_l1024_f20_fs100_ps0_predictions_4",
    ],
    "Julia": [
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_0",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_1",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_2",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_3",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_4",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_5",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_6",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_7",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_8",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_9",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_10",
        "PolyCoder_CodeShopJulia_l1024_f10_fs500_ps0_predictions_11",
        "PolyCoder_CodeShopJulia_l1024_f20_fs0_ps0_predictions_0",
        "PolyCoder_CodeShopJulia_l1024_f20_fs0_ps0_predictions_1",
        "PolyCoder_CodeShopJulia_l1024_f20_fs0_ps0_predictions_2",
        "PolyCoder_CodeShopJulia_l1024_f20_fs0_ps0_predictions_3",
        "PolyCoder_CodeShopJulia_l1024_f20_fs0_ps0_predictions_4",
    ],
}
data = {}
for lang in languages:
    lang_data = None
    for name in files[lang]:
        name = f"./Predictions/{model_shortname}/CodeShop{lang}/{name}"
        dataset_init = datasets.load_dataset(name, split="train")
        if lang_data == None:
            lang_data = dataset_init
        else:
            lang_data = datasets.concatenate_datasets([lang_data, dataset_init])
    data[lang] = lang_data
    combined_name = f"./Predictions/{model_shortname}/CombinedPredictions{lang}/CombinedPredictions{lang}.parquet"
    lang_data.to_parquet(combined_name)

    print(f"{lang} samples: {len(data[lang])}", flush=True)
    