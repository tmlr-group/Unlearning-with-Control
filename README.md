# Towards Effective Evaluations and Comparison for LLM Unlearning Methods

This is the code for the paper [**Towards Effective Evaluations and Comparison for LLM Unlearning Methods**](https://arxiv.org/abs/2406.09179)

## Installation

```
conda create -n unlearning python=3.10
conda activate unlearning
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Loading the Dataset

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU","full")
```

## Finetune your models

The code currently supports `Phi-1.5`, and `Llama2-7b chat` models. But newer models can directly be added in the `model_config.yaml` file. For the unlearning challenege, we fine-tuned `Phi-1.5` for 5 epochs using a maximum learning rate of `2e-5`, and the `Llama2-7b chat` model for the same duration at `1e-5`. Finetuning can be done as follows:

```
master_port=18765
split=full
model=phi #you can choose phi, llama2-7b
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Forget models
Make sure that the path of the model to be unlearned is correctly provided in the `config/model_config.yaml` file. To unlearn a model on a forget set, use the following command:
```
cuda_id=0
master_port=18765
save_steps=25  # save the checkpoint every 25 steps
model=phi #you can choose phi, llama2-7b
forget_loss=ga    # you can choose ga, ga_kl, gd, KL 
CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss} save_steps=$save_steps
```

## Evaluate models
Once you have the model trained, you can generate the PS-series metrics used for evaluation with the following command:

```
cuda_id=0
master_port=18765
model=phi #you can choose phi, llama2-7b
ckpt=baseline/llama2-7b/grad_ascent_1e-05_forget05_8_0.0_250/checkpoint-125  # where the checkpoint is stored
CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$master_port evaluation_everything.py split=${split} model_family=${model} model_path=${ckpt}
```

### Available forget sets are:

- `forget01`: Forgetting 1% of the original dataset, all entries correspond to a single author.
- `forget05`: Forgetting 5% of the original dataset, all entries correspond to a single author.
- `forget10`: Forgetting 10% of the original dataset, all entries correspond to a single author.

Retain sets corresponding to each forget set are also available, which can be used to train an Oracle model.

## Citing Our Work

If you find our metrics beneficial, please cite our work:
```
@inproceedings{wang2025towards,
title={Towards Effective Evaluations and Comparison for LLM Unlearning Methods}, 
author={Qizhou Wang and Bo Han and Puning Yang and Jianing Zhu and Tongliang Liu and Masashi Sugiyama},
booktitle = {International Conference on Learning Representations},
year = {2025}
}
```
