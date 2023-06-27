print("starting processing_tokenize.py")
import datasets
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from transformers import AutoTokenizer
from huggingface_hub import login

print("finished imports")

huggingface_token_filename = "huggingface_token.txt"
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

tokenizers = {
    "PolyCoder": 'NinedayWang/PolyCoder-0.4B',
    "CodeGPT": 'AISE-TUDelft/CodeGPT-Multilingual',
    "CodeParrot": 'codeparrot/codeparrot-small-multi',
    "CodeGen": 'Salesforce/codegen-350M-multi',
}
dataset_names = [
    "CodeShopJavaNoComments",
    "CodeShopGoNoComments",
    "CodeShopKotlinNoComments",
    "CodeShopCPPNoComments",
    "CodeShopJuliaNoComments",
    "CodeShopPythonNoComments",
]
for dataset_name in dataset_names:
    print(f"starting to tokenize {dataset_name}")
    # Load full dataset into memory
    dataset_init = datasets.load_dataset(dataset_name, split="train")
    for k,v in tokenizers.items():
        print(f"starting with {k} tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(v)
        
        data_length_filtered = {
            'id' : [],
            'stripped_content': [],
            'original_content': [],
            'tokenized': []
        }

        for sample in dataset_init:
            tokenized_sample = tokenizer.encode(
                sample['stripped_content'],
                return_tensors="pt"
            )
            tokens = torch.Tensor.tolist(tokenized_sample)[0]
            data_length_filtered['id'].append(sample['id'])
            data_length_filtered['stripped_content'].append(sample['stripped_content'])
            data_length_filtered['original_content'].append(sample['original_content'])
            data_length_filtered['tokenized'].append(tokens)
        
        table = pa.Table.from_pydict(data_length_filtered)
        name = dataset_name + f"{k}Tokenized"
        pq.write_to_dataset(table, root_path=name)
    