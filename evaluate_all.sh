#!/usr/bin/zsh


CMDNAME=`basename $0`


datasets=("CAVE")
concat="False"
model_name=("HSCNN DeepSSPrior HyperReconNet Ghost")
block_num=9
modes=("normal mix1 mix2")
ratios=(2 3 4)


model_name=( `echo $model_name | tr ' ' ' '` )
datasets=( `echo $datasets | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        if [ $name = "Ghost" ]; then
            for ratio in $ratios[@]; do
                for mode in $modes[@]; do
                    python evaluate_reconst_sh.py -d $dataset -c $concat -m $name -b $block_num -r $ratio -md $mode
                done
            done
        else
            python evaluate_reconst_sh.py -d $dataset -c $concat -m $name -b $block_num
        fi
    done
done
