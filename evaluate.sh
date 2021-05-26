#!/usr/bin/zsh


CMDNAME=`basename $0`


datasets="Harvard"
concat="False"
model_name=("HSCNN" "HyperReconNet" "DeepSSPrior" "Ghost")
block_num=9
ratios=(2 3 4)
modes=("normal mix1 mix2 mix3 mix4")
loss_modes=("mse" "mse_sam")
start_time="0521"


while getopts d:c:m:b: OPT
do
    echo "$OPTARG"
    case $OPT in
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        b) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $dataset
echo $concat
echo $block_num


model_name=( `echo $model_name | tr ' ' ' '` )
datasets=( `echo $datasets | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for dataset in $datasets[@]; do
    echo $dataset
    for name in $model_name[@]; do
        if [ $name = "Ghost" ]; then
            for loss_mode in $loss_modes; do

                echo $dataset $name $loss_mode


                for ratio in $ratios[@]; do
                    for mode in $modes[@]; do
                        echo $ratio $mode $name
                        python evaluate_reconst_sh.py -d $dataset -c $concat -m $name -b $block_num -r $ratio -md $mode -st $start_time -l $loss_mode
                    done
                done

            done
        else
            echo $name
            python evaluate_reconst_sh.py -d $dataset -c $concat -m $name -b $block_num -st $start_time -l $loss_mode
        fi
    done
done
