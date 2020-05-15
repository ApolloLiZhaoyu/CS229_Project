#!/usr/bin/env bash

function runexp {

export GLUE_DIR=glue_data
export TASK_NAME=${1}

gpu=${2}      # The GPU you want to use
mname=${3}    # Model name
alr=${4}      # Step size of gradient ascent
amag=${5}     # Magnitude of initial (adversarial?) perturbation
anorm=${6}    # Maximum norm of adversarial perturbation
asteps=${7}   # Number of gradient ascent steps for the adversary
lr=${8}       # Learning rate for model parameters
bsize=${9}    # Batch size
gas=${10}     # Gradient accumulation. bsize * gas = effective batch size
seqlen=512    # Maximum sequence length
hdp=${11}     # Hidden layer dropouts for ALBERT
adp=${12}     # Attention dropouts for ALBERT
ts=${13}      # Number of training steps (counted as parameter updates)
ws=${14}      # Learning rate warm-up steps
seed=${15}    # Seed for randomness
wd=${16}      # Weight decay

expname=FreeLB-${mname}-${TASK_NAME}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-hdp${hdp}-adp${adp}-ts${ts}-ws${ws}-wd${wd}-seed${seed}

python examples/run_glue_freelb.py \
  --model_type albert \
  --model_name_or_path ${mname} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length ${seqlen} \
  --per_gpu_train_batch_size ${bsize} --gradient_accumulation_steps ${gas} \
  --learning_rate ${lr} --weight_decay ${wd} \
  --gpu ${gpu} \
  --output_dir checkpoints/${expname}/ \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
  --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
  --expname ${expname} --evaluate_during_training \
  --max_steps ${ts} --warmup_steps ${ws} --seed ${seed} \
  --logging_steps 1000 --save_steps 1000 \
  --comet \
  > logs/${expname}.log 2>&1 &
}


# runexp TASK_NAME  gpu    model_name          adv_lr    adv_mag    anorm    asteps    lr       bsize    grad_accu   hdp   adp    ts     ws       seed      wd
runexp   CoLA       0      albert-xxlarge-v2   2.5e-2    4e-1       3e-1     3         1e-5     16       1           0     0      5536   320      9017      0.1
runexp   CoLA       1      albert-xxlarge-v2   2.5e-2    4e-1       3e-1     3         1e-5     16       1           0     0      5536   320      1125      0.01
runexp   CoLA       2      albert-xxlarge-v2   2.5e-2    4e-1       3e-1     3         1e-5     16       1           0     0      5536   320      42        0.01
