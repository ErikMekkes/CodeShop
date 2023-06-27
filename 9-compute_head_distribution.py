import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

input_size = 1024

summed_heads = datasets.load_dataset(f"./SummedHeads/{input_size}", split="train")["summed_heads"]

b = pd.DataFrame(summed_heads)
head_distrib = b.describe()
print(head_distrib)

#                0             1             2             3     ...          1020          1021          1022          1023
# count  11952.000000  11952.000000  11952.000000  11952.000000  ...  1.195200e+04  11952.000000  1.195200e+04  1.195200e+04
# mean     242.073009      2.866936      2.527238      2.394642  ...  1.505083e-01      0.127494  9.395281e-02  2.963245e-02
# std      183.394977      3.120850      2.864595      2.640299  ...  2.651619e-01      0.235043  1.806746e-01  5.116121e-02
# min        1.076210      0.015172      0.000381      0.000365  ...  3.330019e-07      0.000002  6.577584e-09  1.780424e-08
# 25%       77.917169      0.920438      0.840064      0.831330  ...  1.023994e-02      0.007877  5.690007e-03  2.313479e-03
# 50%      225.505115      1.773513      1.578854      1.548236  ...  4.074882e-02      0.032589  2.397462e-02  9.648363e-03
# 75%      369.340378      3.631279      3.138299      3.026508  ...  1.663518e-01      0.130664  9.279709e-02  3.506353e-02
# max      979.735414     40.557599     46.504153     52.275163  ...  2.609478e+00      2.008372  1.237565e+00  7.498066e-01

# stats : [0:count, 1:mean, 2:std, 3:min, 4:25%, 5:median, 6:75%, 7:max]
# [token_i][stat_i]

num_tokens = len(summed_heads[0])
num_samples = len(summed_heads)
print("num_samples: ", num_samples)
print("num_tokens: ", num_tokens)

# [head_i][token_i][score] -> [token_i][head_i][score]
token_indexed_summed_heads = []
for token_i in range(num_tokens):
    token_scores = []
    for head in summed_heads:
        token_scores.append(head[token_i])
    token_indexed_summed_heads.append(token_scores)

sorted_token_indexed_sums = []
for token_i in range(num_tokens):
    sorted_token_indexed_sums.append(sorted(token_indexed_summed_heads[token_i]))

data_dict = {
    "head_distribution" : sorted_token_indexed_sums
}
table = pa.Table.from_pydict(data_dict)
name = f"./HeadDistribution/{input_size}"
pq.write_to_dataset(table, root_path=name)

def get_perct(scores, perct):
    n = len(scores)
    index = perct*(n-1)
    q, mod = divmod(index, 1)
    if mod == 0:
        return scores[int(q)]
    else:
        return (scores[int(q)]+mod*(scores[int(q+1)]-scores[int(q)]))

# print(get_perct([1,2,3,4,5,6,7,8,9,10], 0.25))
# print(get_perct(sorted_token_indexed_sums[0],0.25))
# print(get_perct(sorted_token_indexed_sums[0],0.5))
# print(get_perct(sorted_token_indexed_sums[0],0.75))

