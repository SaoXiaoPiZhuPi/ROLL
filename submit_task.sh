#!/bin/bash
#algo_name 用来指定pytorch版本号,目前支持pytorch180、pytorch1110、pytorch1131、pytorch201

args="--key=value \
 --key1=value1  \
--last-conv-stride=2"
echo "${args}"

# 可选 ENVS，每个环境变量之间用逗号分割，K1=V1,K2=V2
# 不支持环境变量V里面包含逗号,
# ENVS="NVTE_BWD_LAYERNORM_SM_MARGIN=4,NCCL_IB_QPS_PER_CONNECTION=4,CUDA_DEVICE_MAX_CONNECTIONS=1"
echo $ENVS
nebulactl run mdl --queue= \
                  --entry=train.py \
                  --algo_name=pytorch180 \
                  --worker_count=8 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --tables=your_input_odps_tables \   可选
                  --job_name=your_job_name \    可选
                  --max_failover_times=0 \   可选，训练任务无需配置
                  --env=${ENV} \可选
                  
                  
