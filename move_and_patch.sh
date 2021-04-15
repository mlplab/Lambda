#!/usr/bin/zsh


CMDNAME=`basename $0`


dataset=("CAVE" "Harvard" "ICVL")


# python move_and_patch_h5.py 


for name in $dataset[@]; do
    cd "../SCI_dataset/"
    pwd
    tar -zxvf "../SCI_dataset/${name}.tar.gz"
    cd "../deep_SS_prior/"
    pwd
    python move_and_patch.py -d $name
done

