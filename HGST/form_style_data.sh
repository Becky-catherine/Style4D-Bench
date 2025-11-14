#!/bin/bash


for i in $(seq -f "cam%02g" 0 20)
do
    
    mkdir -p "$i"
    
   
    python test_video_style.py \
        --content_dir "4DGaussians/data/dynerf/flame_steak/$i/images" \
        --style_dir "st2" \
        --output "$i"
    
    echo "$i finished!"
done
