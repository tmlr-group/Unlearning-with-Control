import os
import subprocess
import argparse
import re
import os, random, argparse, time, subprocess

def find_checkpoint_directories(root_path):
 
    checkpoint_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if dirname.startswith('checkpoint-'):
                full_path = os.path.join(dirpath, dirname)
                checkpoint_dirs.append(full_path)
    return checkpoint_dirs


def execute_command(cuda_id, model_family, split, model_path):
    command = (
        f"CUDA_VISIBLE_DEVICES={cuda_id} "
        f"torchrun --nproc_per_node=1 "
        f"--master_port={random.randint(10000, 60000)} "
        f"evaluation_everything.py "
        f"model_family={model_family} split={split} model_path={model_path}"
    )
    print("Executing command:", command)
    subprocess.run(command, shell=True)



def main():
    parser = argparse.ArgumentParser(description="Automatically run evaluation on checkpoints.")
    parser.add_argument('--root_path', type=str, required=True, help='Root directory to search for checkpoint folders.')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID.')
  
    args = parser.parse_args()

    checkpoint_dirs = find_checkpoint_directories(args.root_path)
    if not checkpoint_dirs:
        print("No checkpoint directories found.")
        return

    for checkpoint_dir in checkpoint_dirs:
        tokens = checkpoint_dir.split("/")

        if 'forget01' in tokens[-2]:
            split = 'forget01'

        elif 'forget05' in tokens[-2]:
            split = 'forget05'

        elif 'forget10' in tokens[-2]:
            split = 'forget10'

        print("executing:",tokens[-2])

        model = tokens[-3]
        execute_command(args.cuda_id, model, split, checkpoint_dir)

if __name__ == "__main__":
    main()