#!/usr/bin/zsh


CMDNAME=`basename $0`


dataset="Harvard"
concat="False"
model_name=("HSCNN DeepSSPrior HyperReconNet")
block_num=9


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
for name in $model_name[@]; do
    echo $name
done
for name in $model_name[@]; do
    python evaluate_reconst_sh.py -d $dataset -c $concat -m $name -b $block_num
done
