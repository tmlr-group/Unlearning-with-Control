# Unlearning with Control: Assessing Real-world Utility for LLM Unlearning 

This is the code for the paper [**Unlearning with Control: Assessing Real-world Utility for LLM Unlearning**](https://arxiv.org/abs/2406.09179)

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
model=phi
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Forget models
Make sure that the path of the model to be unlearned is correctly provided in the `config/model_config.yaml` file. To unlearn a model on a forget set, use the following command:
```
save_steps=25  # save the checkpoint every 25 steps
forget_loss=grad_diff    # you can choose grad_diff, grad_ascent, ocr_ce, only_fgt 
delta=6 #this is the upper bound of unlearning control
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_2.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss} save_steps=${save_steps} delta_ocr=${delta}
```

## Evaluate models
Once you have the model trained, you can generate the PS-series merics used for evaluation with the following command:

For llama:

```
ckpt=baseline/llama2-7b/grad_ascent_1e-05_forget05_8_0.0_250/checkpoint-125  # where the checkpoint is stored
update_rate=0.0 # the ratio of mixing the unlearned model and the original model, 0.0 means totally unlearned model
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 llama_exact.py model_family=llama2-7b split=${split} model_path=${ckpt} update_=0.0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 llama_similar.py model_family=llama2-7b split=${split} model_path=${ckpt} update_=0.0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 llama_perturb.py model_family=llama2-7b split=${split} model_path=${ckpt} update_=0.0
```

For phi:

```
ckpt=baseline/phi/grad_ascent_2e-05_forget05_8_0.0_250/checkpoint-125  # where the checkpoint is stored
update_rate=0.0 # the ratio of mixing the unlearned model and the original model, 0.0 means totally unlearned model
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 phi_exact.py model_family=phi split=${split} model_path=${ckpt} update_=0.0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 phi_similar.py model_family=phi split=${split} model_path=${ckpt} update_=0.0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18141 lphi_perturb.py model_family=phi split=${split} model_path=${ckpt} update_=0.0
```
### Available forget sets are:

- `forget01`: Forgetting 1% of the original dataset, all entries correspond to a single author.
- `forget05`: Forgetting 5% of the original dataset, all entries correspond to a single author.
- `forget10`: Forgetting 10% of the original dataset, all entries correspond to a single author.

Retain sets corresponding to each forget set are also available, which can be used to train an Oracle model.

## Citing Our Work

If you find our metrics beneficial, please cite our work:
```
@article{wang2024unlearning,
  title={Unlearning with Control: Assessing Real-world Utility for Large Language Model Unlearning},
  author={Wang, Qizhou and Han, Bo and Yang, Puning and Zhu, Jianing and Liu, Tongliang and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2406.09179},
  year={2024}
}
```
## Quick Links
This project is heavily rely on the [**TOFU**](https://github.com/locuslab/tofu)