#!/usr/bin/env zsh
# /usr/bin/env17 Bruno Ribeiro, Mayank Kakodkar, Pedro Savarese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


GPU="$1"
echo "GPU=$GPU"
GPULIMIT="$2"
echo "GPULIMIT=$GPULIMIT"
wd="$3" # "0.01","0.001","0.0001","0.00001"
echo "wd=$wd"

BASE_DIR="/homes/mkakodka/r/MCLV-RBM/"

for lr in {"0.001","0.01","0.1","1.0"}
 do
    for method in {"CD","PCD"}
     do
         for cdk in "1"
             do
                export CUDA_VISIBLE_DEVICES="$GPU"
                python3 "$BASE_DIR"py/main.py -b "$BASE_DIR" -n 10 --method $method -cdk $cdk -tot 100 --plateau 10 --hidden 25 -lr $lr -wd $wd --final-likelihood --filename h25-$method-$lr-$wd-$cdk --gpu-limit $GPULIMIT
             done
     done
     for mclvk in "1"
      do
        export CUDA_VISIBLE_DEVICES="$GPU"
        python3 "$BASE_DIR"py/main.py -b "$BASE_DIR" -n 10 --method MCLV -cdk 1 -mclvk $mclvk -tot 100 -wm 15 --plateau 10 --hidden 25 -lr $lr -wd $wd --final-likelihood --filename h25-MCLV-$lr-$wd-$mclvk --gpu-limit $GPULIMIT
       done
 done
