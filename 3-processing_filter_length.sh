#!/bin/bash
#SBATCH --job-name=filter_length
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2022r2 miniconda3

export HF_DATASETS_CACHE="/scratch/emekkes/hf_datasets_cache"

cd /scratch/emekkes

conda activate codeshop_cuda
python processing_filter_length.py > processing_filter_length.log
conda deactivate
