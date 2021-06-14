#!/usr/bin/zsh


CMDNAME=`basename $0`


dataset=("CAVE" "Harvard" "ICVL")
patch_size=(48 64 128 256)


# python move_and_patch_h5.py 


for name in $dataset[@]; do
    for patch in $patch_size[@]; do
        echo $name $patch
        # cd "../SCI_dataset/"
        # pwd
        # tar -zxvf "../SCI_dataset/${name}.tar.gz"
        # cd "../Deep_SS_Prior/"
        # pwd
        python move_and_patch.py -d $name -p $patch
    done
done

