#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
datasets=("CAVE Harvard")
concat="False"
model_name=("Ghost")
block_num=9
ratios=(2 3 4)
modes=("normal mix1 mix2 mix3 mix4")
loss_modes=("mse" "mse_sam")
start_time=$(date "+%m%d")


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) datasets=$OPTARG ;;
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
echo $start_time


model_name=( `echo $model_name | tr ' ' ' '` )
datasets=( `echo $datasets | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        for loss_mode in $loss_modes; do

            echo $dataset $name $loss_mode

            if [ $name = "Ghost" ]; then

                for ratio in $ratios[@]; do
                    for mode in $modes[@]; do
                        echo $ratio $mode $name
                        python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -r $ratio -md $mode -st $start_time -l $loss_mode
                    done
                done

            else
                echo $name 
                python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -st $start_time -l $loss_mode
            fi

        done
    done
done
