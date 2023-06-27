print("starting processing_strip_comments.py")

import datasets
from utilities import (
    removeCommentsCpp,
    removeCommentsJavaKotlinGo,
    removeCommentsJulia,
    removeCommentsPython
)
from huggingface_hub import login
import pyarrow.parquet as pq
import pyarrow as pa

print("finished imports")

huggingface_token_filename = "huggingface_token.txt"
huggingface_token_file = open(huggingface_token_filename, "r")
auth_token = huggingface_token_file.read()
huggingface_token_file.close()
login(auth_token)

# dataset_names = [
#     "CodeShopJava",
#     "CodeShopGo",
#     "CodeShopKotlin",
#     "CodeShopCPP",
#     "CodeShopJulia",
#     "CodeShopPython",
# ]
dataset_names = [
    "AISE-TUDelft/CodeShopJava",
    "AISE-TUDelft/CodeShopGo",
    "AISE-TUDelft/CodeShopKotlin",
    "AISE-TUDelft/CodeShopCPP",
    "AISE-TUDelft/CodeShopJulia",
    "AISE-TUDelft/CodeShopPython",
]
for dataset_name in dataset_names:
    print(f"starting to filter comments from {dataset_name}")
    # Load full dataset into memory
    dataset_init = datasets.load_dataset(dataset_name, split='train')

    data_stripped_comments = {
        'id' : [],
        'stripped_content': [],
        'original_content': [],
    }
    # This should take at most 2 mins 
    for row in dataset_init:
        if dataset_name == 'CodeShopJava' or dataset_name == 'CodeShopKotlin' or dataset_name == 'CodeShopGo':
            filter_func = removeCommentsJavaKotlinGo
        elif dataset_name == 'CodeShopPython':
            filter_func = removeCommentsPython
        elif dataset_name == 'CodeShopJulia':
            filter_func = removeCommentsJulia
        else:
            filter_func = removeCommentsCpp
        
        newRow = filter_func(row['content'])
        data_stripped_comments['id'].append(row['id'])
        data_stripped_comments['stripped_content'].append(newRow)
        data_stripped_comments['original_content'].append(row['content'])
    

    table = pa.Table.from_pydict(data_stripped_comments)
    name = dataset_name.split("/")[-1] + "NoComments"
    pq.write_to_dataset(table, root_path=name)