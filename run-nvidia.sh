#!/bin/bash

image="/scratch/aolpin/thesis/data/1_2016-11-15-11_00.bmp"
output="/scratch/aolpin/thesis/results/"
model="/scratch/aolpin/thesis/model/"

date_time=`date +%Y-%m-%d--%H-%M-%S-%N`

python usemodel.py $image "${output}cnn" "${model}cnn/2017-10-24--00-20-08-057174142" "True"
python usemodel.py $image "${output}fcn" "${model}fcn/2017-10-25--01-16-07-143592620" "False"