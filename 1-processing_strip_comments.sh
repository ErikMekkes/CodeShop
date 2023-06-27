#!/bin/bash
#SBATCH --job-name=strip_comments
#SBATCH --partition=compute
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2022r2 miniconda3

export HF_DATASETS_CACHE="/scratch/emekkes/hf_datasets_cache"

cd /scratch/emekkes

conda activate codeshop_cuda
python processing_strip_comments.py > processing_strip_comments.log
conda deactivate
