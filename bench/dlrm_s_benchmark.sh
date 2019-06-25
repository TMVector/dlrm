#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

build=0
cpu=1
gpu=1
pt=1
c2=1

ncores=12
nsockets="0"

ngpus="1"

numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

data=random #synthetic
print_freq=100
rand_seed=727

c2_net="async_scheduling"

### Model param
################################################################################

# The Architectural Implications of Facebook’s
# DNN-based Personalized Recommendation
# https://arxiv.org/pdf/1906.03109.pdf

# Section 2.3
#
# The number of embedding tables varies from 4 to 40.
#
# In aggregate, embedding tables for a single
# recommendation model can consume up to 20GB of memory.

# Section 3
#
# "In this section we describe model architectures for three
# classes of production-scale recommendation models, referred
# to as RM1, RM2, and RM3. The three model types are used
# across two different services and have *different configurations*
# based on their use-case." As many variants of each type
# of model exist across production-scale recommendation systems,
# we provide a range of parameters for RM1, RM2, and
# RM3.
#
# RM1 is a lightweight recommendation model used in the filtering
# step when higher accuracy is needed (rather than logistic regression).
#
# RM3 is a heavyweight recommendation model, used for ranking social media posts.
# It comprises of larger Bottom-FC layers. This is a result of the service using more dense features.
#
# RM2 is a heavyweight recommendation model, but it comprises of more
# embedding tables as it processes contents with more sparse features
#
# Each query must be processed within strict latency constraints set by SLA.
# Based on the use case, the SLA requirements can vary from tens to hundreds of
# milliseconds.
#
# In the data center, balancing throughput with strict latency requirements
# is accomplished by batching queries and co-locating multiple
# inferences on the same machine.

### Section 5.1 Configuring the open-source benchmark
#
# "Example configurations for RM1, RM2, and RM3: As
# an example on how to configure the open-source benchmark
# to represent production scale recommendation workloads,
# lets consider a RM1 model shown in Table 1. In this model
# the number of embedding tables can be set to 5, with input
# and output dimensions of 10^5
# and 32, the number of sparse
# lookups to 80, depth and width of Bottom-MLP layers to
# 3 and 128−64−32, and the depth and width of Top-MLP
# layers to 3 and 128 − 32 − 1. The RM2 and RM3 models
# have been configured similarly."

# Table 1
# +-------+-----------+----------+-----------+------------+-------------+-----------------------+
# | Model | bot_mlp   | top_mlp  | emb_num   | emb_in_dim | emb_out_dim | emb_lookups           |
# +-------+-----------+----------+-----------+------------+-------------+-----------------------+
# | RM1   |  8x/4x/1x | 4x/2x/1x | 1x to  3x | 1x to 180x | 1x          | User: 4x, posts: Nx4x |
# | RM2   |  8x/4x/1x | 4x/2x/1x | 8x to 12x | 1x to 180x | 1x          | User: 4x, posts: Nx4x |
# | RM3   | 80x/8x/4x | 4x/2x/1x | 1x to  3x | 1x to 180x | 1x          | 1x                    |
# +-------+-----------+----------+-----------+------------+-------------+-----------------------+
# Each parameter (column) is normalized to the smallest instance. For example, Bottom and Top FC sizes are normalized
# to layer 3 in RM1. Number, input dimension, and output dimension of embedding tables are normalized to the RM1
# model. Number of lookups are normalized to RM3.

# These two don't quite add up for RM1, as 128-64-32 is 4x/2x/1x for the bottom, and 128-32-1 is 128x/32x/1x for the top...

mb_size=256 #2048 #1024 #512 #256
nbatches=1000 #500 #100
interaction="dot"

# Parameters:
# bot_mlp    # Layer widths of bottom MLP
# top_mlp    # Layer widths of top MLP
# emb_size   # Embedding vector size (embedding output dimension)
# nindices   # Number of sparse lookups
# emb        # Input sizes of embedding matrices

# RM1
bot_mlp="128-64-32"
top_mlp="128-32-1"
emb_size=32
nindices=80
emb="100000-100000-100000-100000-100000" # Up to 15 tables and up to 18000000 size?

################################################################################

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --inference-only"\
" --enable-profiling "

c2_args=" --caffe2-net-type="${c2_net}

if [ $build = 1 ]; then
  BUCK_DISTCC=0 buck build @mode/opt //experimental/mnaumov/hw/dlrm:dlrm_s_pytorch //experimental/mnaumov/hw/dlrm:dlrm_s_caffe2
fi

# CPU Benchmarking
if [ $cpu = 1 ]; then
  echo "--------------------------------------------"
  echo "CPU Benchmarking - running on $ncores cores"
  echo "--------------------------------------------"
  if [ $pt = 1 ]; then
    outf="model1_CPU_PT_$ncores.log"
    outp="dlrm_s_pytorch.prof"
    echo "-------------------------------"
    echo "Running PT (log file: $outf)"
    echo "-------------------------------"
    cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size $_args $dlrm_extra_option > $outf"
    echo $cmd
    eval $cmd
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # move profiling file(s)
    mv $outp ${outf//".log"/".prof"}
    mv ${outp//".prof"/".json"} ${outf//".log"/".json"}

  fi
  if [ $c2 = 1 ]; then
    outf="model1_CPU_C2_$ncores.log"
    outp="dlrm_s_caffe2.prof"
    echo "-------------------------------"
    echo "Running C2 (log file: $outf)"
    echo "-------------------------------"
    cmd="$numa_cmd $dlrm_c2_bin --mini-batch-size=$mb_size $_args $c2_args $dlrm_extra_option 1> $outf 2> $outp"
    echo $cmd
    eval $cmd
    min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
    echo "Min time per iteration = $min"
    # move profiling file (collected from stderr above)
    mv $outp ${outf//".log"/".prof"}
  fi
fi

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    if [ $pt = 1 ]; then
      outf="model1_GPU_PT_$_ng.log"
      outp="dlrm_s_pytorch.prof"
      echo "-------------------------------"
      echo "Running PT (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size $_args --use-gpu $dlrm_extra_option > $outf"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file(s)
      mv $outp ${outf//".log"/".prof"}
      mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    fi
    if [ $c2 = 1 ]; then
      outf="model1_GPU_C2_$_ng.log"
      outp="dlrm_s_caffe2.prof"
      echo "-------------------------------"
      echo "Running C2 (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_c2_bin --mini-batch-size=$_mb_size $_args $c2_args --use-gpu $dlrm_extra_option 1> $outf 2> $outp"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file (collected from stderr above)
      mv $outp ${outf//".log"/".prof"}
    fi
  done
fi
