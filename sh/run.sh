#!/usr/bin/env bash
nhup "python3 $rbm_base --base-folder $rbm_base_folder \
--phase DATA \
--runs 3 \
--total-epochs 20 \
--warmup-epochs 2 \
--num-hidden 25 \
--method MCLV CD \
--k 1 \
--batch-size 128 \
--learning-rate 0.1 \
--schedule RM100  \
--gpu-id 0 2 3 \
--log-tour \
" "BRUNO"