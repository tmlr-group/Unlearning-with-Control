import pdb, os, hydra
import logging
import random,time
import numpy as np
import sklearn.metrics as sk

import torch
from torch.nn import Softmax
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from utils import get_model_identifiers_from_yaml
from data_module import model_mix

import safetensors
log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def generate(model,tokenizer,question_prompt):
    inputs = tokenizer.batch_encode_plus(question_prompt, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=150, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    strs = tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return strs


@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    before_ckpt=safetensors.torch.load_file('data/weight/ft_epoch5_lr2e-05_phi_full_wd0.0/checkpoint-625/model.safetensors')
    #before_ckpt_1=safetensors.torch.load_file('data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00001-of-00003.safetensors')
    #before_ckpt_2=safetensors.torch.load_file('data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00002-of-00003.safetensors')
    #before_ckpt_3=safetensors.torch.load_file('data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00003-of-00003.safetensors')
    #before_ckpt={**before_ckpt_1,**before_ckpt_2,**before_ckpt_3}
    after_ckpt=safetensors.torch.load_file(cfg.model_path+'/model.safetensors')
    #after_ckpt1=safetensors.torch.load_file(cfg.model_path+'/model-00001-of-00003.safetensors')
    #after_ckpt2=safetensors.torch.load_file(cfg.model_path+'/model-00002-of-00003.safetensors')
    #after_ckpt3=safetensors.torch.load_file(cfg.model_path+'/model-00003-of-00003.safetensors')
    #after_ckpt={**after_ckpt1,**after_ckpt2,**after_ckpt3}
    update_ratio=cfg.update_
    model=model_mix(model,before_ckpt,after_ckpt,update_ratio)
    model = model.eval()

    if cfg.split=='forget10':
        retain_name='retain90_perturbed'
    elif  cfg.split=='forget05':
        retain_name='retain95_perturbed'
    elif  cfg.split=='forget01':
        retain_name='retain99_perturbed'
    retain_eval_data=load_dataset('locuslab/TOFU','retain_perturbed')['train'].train_test_split(train_size=40,shuffle=False)['train']
    if cfg.split=='forget01':
        forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train']
    else:
        forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train'].train_test_split(train_size=40)['train']

    retain_eval_loader=torch.utils.data.DataLoader(retain_eval_data,batch_size=1)
    forget_loader=torch.utils.data.DataLoader(forget_data,batch_size=1)

    log1 = logging.getLogger("Unlearning")
    log_file_path = cfg.model_path+f'/perturbed_mix_{cfg.update_}.log'
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    log1.addHandler(file_handler)

    retain_sum = 0.0
    for i, j in enumerate(retain_eval_loader):
        #print(f'The {i} question:')
        log1.info(f'The {i} question:')
        #log1.critical("question------------------------------")
        #print("question------------------------------")
        question = f"### Question: {j['paraphrased_question'][0]}\n ### Answer:"
        generated_prompt = generate(model, tokenizer, [question])
        if cfg.model_family in ['phi']:
            ans=generated_prompt[0][1:].split('\n')[0]
        else:
            ans=generated_prompt[0][:].split('\n')[0]
        #print(generated_prompt[0].split('\n')[0])
        #print(j['answer'][0])
        #pdb.set_trace()
        #print('adjust:', ans)
        #print('Ground truth:', j['answer'][0])
        #time.sleep(2)
        if ans == j['answer'][0]:
            #print(0.0)
            log1.info('0.0')
            continue
        gt_length=len(j['answer'][0].split(" "))
        for k in range(gt_length):
            #print("question------------------------------")
            question = f"### Question: {j['paraphrased_question'][0]}\n ### Answer: " + " ".join(j['answer'][0].split(" ")[:k + 1])
            generated_prompt = generate(model,tokenizer, [question])
            ans=generated_prompt[0][:].split('\n')[0]
            #print('Answer: ', generated_prompt[0].split('\n')[0])
            #print(f'{k+1} words:')
            #print(type(ans))
            #print(type(j['answer'][0].split(" ")[:k + 1]))
            #print('Rethink: ',' '.join(j['answer'][0].split(" ")[:k + 1])+ans)
            #print('Ground truth: ', j['answer'][0])
            if (' '.join(j['answer'][0].split(" ")[:k + 1])+ans) == j['answer'][0]:
                #print(f"ratio: {(k + 1) / gt_length}")
                log1.info(f"ratio: {(k + 1) / gt_length}")
                retain_sum += ((k + 1) / gt_length)
                break
            elif k==gt_length-1:
                #print(f"other: {1}")
                log1.info(f"other: {1}")
                retain_sum += 1
                break
    #print('Final Results(Retain): ',1 - retain_sum/100)
    log1.info(f"Final Results(Retain): {1 - retain_sum/40}")

    forget_sum = 0.0

    for i, j in enumerate(forget_loader):
        #print(f'The {i} question:')
        log1.info(f'The {i} question:')
        question = f"### Question: {j['paraphrased_question'][0]}\n ### Answer:"
        generated_prompt = generate(model, tokenizer, [question])
        ans=generated_prompt[0][1:].split('\n')[0]
        #time.sleep(2)
        if ans == j['answer'][0]:
            #print(0.0)
            log1.info('0.0')
            continue
        gt_length=len(j['answer'][0].split(" "))
        for k in range(gt_length):
            #print("question------------------------------")
            question = f"### Question: {j['paraphrased_question'][0]}\n ### Answer: " + " ".join(j['answer'][0].split(" ")[:k + 1])
            generated_prompt = generate(model,tokenizer, [question])
            ans=generated_prompt[0][:].split('\n')[0]
            #print('Answer: ', generated_prompt[0].split('\n')[0])
            #print(f'{k+1} words:')
            #print(type(ans))
            #print(type(j['answer'][0].split(" ")[:k + 1]))
            #print('Rethink: ',' '.join(j['answer'][0].split(" ")[:k + 1])+ans)
            #print('Ground truth: ', j['answer'][0])
            if (' '.join(j['answer'][0].split(" ")[:k + 1])+ans) == j['answer'][0]:
                #print(f"ratio: {(k + 1) / gt_length}")
                log1.info(f"ratio: {(k + 1) / gt_length}")
                forget_sum += ((k + 1) / gt_length)
                break
            elif k==gt_length-1:
                #print(f"other: {1}")
                log1.info(f"other: {1}")
                forget_sum += 1
                break
    #print('Final Results(Forget): ', 1-forget_sum/100)
    log1.info(f"Final Results(Forget): {1 - forget_sum/40}")


if __name__ == "__main__":
    main()