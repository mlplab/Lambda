#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
dataset="Harvard"
concat="False"
model_name="Lambda"
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


python reconst.py -st $start_time -d $dataset
python refine.py -st $start_time -d $dataset
