#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
datasets="CAVE"
concat="False"
model_name=("HSCNN DeepSSPrior HyperReconNet Ghost")
block_num=9
ratios=(2 3 4)
modes=("mix1 mix2")


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $batch_size
echo $epoch
echo $dataset
echo $block_num


model_name=( `echo $model_name | tr ' ' ' '` )
dataset=( `echo $datasets | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        if [ $name = "Ghost" ]; then
            for ratio in $ratios[@]; do
                for mode in $modes[@]; do
                    echo $mode
                    python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -r $ratio -md $mode
                done
            done
        else
            python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num
        fi
    done
done
