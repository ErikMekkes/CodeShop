print("starting processing_filter_length.py")

import datasets
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import login
import numpy as np

print("finished imports")

huggingface_token_filename = "huggingface_token.txt"
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

# fewer than this = poor predictions
min_left_context = 1024
# how many predictions do we want in the section
section_length = 1000

# this filtering results in all samples having an identical token length:
#    min_left_context + section_length
# if a sample has less tokens, it is excluded.
# if a sample has more tokens, a random subsection is taken.

tokenizers = {
    "PolyCoder": "NinedayWang/PolyCoder-0.4B",
    "CodeGPT": "AISE-TUDelft/CodeGPT-Multilingual",
    "CodeParrot": "codeparrot/codeparrot-small-multi",
    "CodeGen": "Salesforce/codegen-350M-multi",
}
dataset_names = [
    "CodeShopJava",
    "CodeShopGo",
    "CodeShopKotlin",
    "CodeShopCPP",
    "CodeShopJulia",
    "CodeShopPython",
]
for dataset_name in dataset_names:
    print(f"starting to filter by minimum length {dataset_name}")

    for k,v in tokenizers.items():
        print(f"starting with {k} tokenized files")
        # Load full dataset into memory
        dataset_init = datasets.load_from_disk(f"./PreProcessed/Tokenized/{dataset_name}NoComments{k}Tokenized")
        
        data_length_filtered = {
            "id" : [],
            "original_content": [],
            "stripped_content": [],
            "tokenized": [],
            "tokenized_section": [],
            "subsection_boolean": [],
        }
        n = 0
        for sample in dataset_init:
            sample_id = sample["id"]
            sample_tokens = sample["tokenized"]
            sample_length = len(sample_tokens)
            np.random.seed(sample["id"])

            # fix for random start pos cant be 0 due to implementation of
            #    randomInt(0, x): x must be > 0
            if sample_length < min_left_context + section_length:
                continue
            if sample_length == min_left_context + section_length:
                start = 0
                subsection = False
            else:
                start = np.random.randint(
                    0,
                    sample_length-min_left_context-section_length
                )
                subsection = True
            end = start + min_left_context + section_length
            data_length_filtered["id"].append(sample["id"])
            data_length_filtered["original_content"].append(sample["original_content"])
            data_length_filtered["stripped_content"].append(sample["stripped_content"])
            data_length_filtered["tokenized"].append(sample_tokens)
            data_length_filtered["tokenized_section"].append(sample_tokens[start:end])
            data_length_filtered["subsection_boolean"].append(subsection)
        
        table = pa.Table.from_pydict(data_length_filtered)
        name = f"./test/{k}/{dataset_name[:-10]}{k}PreProcessed"
        pq.write_to_dataset(table, root_path=name)