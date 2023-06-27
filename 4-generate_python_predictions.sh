#!/bin/bash
#SBATCH --job-name=python_predictions
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --account=education-eemcs-courses-cse3000


module load 2022r2 miniconda3

export HF_DATASETS_CACHE="/scratch/emekkes/hf_datasets_cache"

cd /scratch/emekkes/CodeShop

conda activate codeshop_cuda
python generate_predictions.py --language="Python" --model="NinedayWang/PolyCoder-0.4B" --lens="AISE-TUDelft/PolyCoder-lens" --files=500 --file_start_index=100 --batch_size=20 --split_size=20 --input_size=1024 --pred_start_index=0 --device="cuda" --huggingface_token="./huggingface_token.txt" --max_runtime=175 > generate_predictions_python.log
conda deactivate
