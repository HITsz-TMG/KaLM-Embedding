

# GPUS_PER_NODE=$(python3 -c 'import torch_npu; print(torch_npu.npu.device_count())')
GPUS_PER_NODE=$(python3 -c 'import torch; print(torch.cuda.device_count())')
WORLD_SIZE=
RANK=
MASTER_ADDR=
MASTER_PORT=


model_name_or_path=
train_data=
output_dir=
ds_config_path=$(dirname "$0")/ds_config_zero-0.json
per_device_train_batch_size=48
query_instruction_for_retrieval=""
passage_instruction_for_retrieval=""
sentence_pooling_method=mean


cd $(dirname "$0")/..

torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m train.run \
    --deepspeed ${ds_config_path} \
    --gradient_checkpointing True \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --train_data ${train_data} \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --bf16 \
    --num_train_epochs 1 \
    --save_strategy "epoch" \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --dataloader_drop_last True \
    --dataloader_num_workers 32 \
    --sentence_pooling_method ${sentence_pooling_method} \
    --normlized True \
    --temperature 0.01 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --train_group_size 8 \
    --negatives_cross_device True \
    --use_expaned_neg False \
    --sample_intask_neg False \
    --sample_intask_neg_ratio 0 \
    --logging_steps 5 \
    --query_instruction_for_retrieval "${query_instruction_for_retrieval}" \
    --passage_instruction_for_retrieval "${passage_instruction_for_retrieval}" \
    --matryoshka_dims 896 512 256 128 64 \
    --matryoshka_weights 1 0.3 0.2 0.1 0.1 \
    --use_matryoshka

