#!/bin/bash
#SBATCH --job-name=train-bart_detection
#SBATCH -t 04:00:00                                         # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared                                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                           # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G                                    # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                                           # total number of nodes
#SBATCH --ntasks=1                                          # total number of tasks
#SBATCH --cpus-per-task=8                                   # number cores per task
#SBATCH --mail-type=all                                     # send mail when job begins and ends
#SBATCH --mail-user=TODO@stud.uni-goettingen.de   
#SBATCH --output=./slurm_files/slurm-%x-%j.out              # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err               # where to write slurm error

source activate dnlp

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

#! /bin/bash
source ~/.bashrc
conda activate dnlp
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

python bart_detection.py --use_gpu --learning_rate 2.8895379286156533e-05 --weight_decay 0.00010623208853699683 --dropout 0.35265700071125883 --classifier_hidden_size 1536 --warmup_ratio 0.10219797504667343 --batch_size 4 --epochs 20 --patience 5 --loss BCE --pooling mean --max_length 512 --seed 11711

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
