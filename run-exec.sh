#!/bin/bash

image="/scratch/aolpin/predata/images/1_2016-11-15-11_00.bmp"
output="/scratch/aolpin/execution/"
model="/scratch/aolpin/thesis/results/"

date_time=`date +%Y-%m-%d--%H-%M-%S-%N`

sqsub -r 4h -q gpu -f threaded -n 4 --gpp=8 --mpp=24g -o "${output}cnn-$date_time.log" python usemodel.py $image "${output}cnn" "${model}cnn/2017-10-24--00-20-08-057174142" "True"
sqsub -r 4h -q gpu -f threaded -n 4 --gpp=8 --mpp=24g -o "${output}fcn-$date_time.log" python usemodel.py $image "${output}fcn" "${model}fcn/2017-10-24--00-20-09-657680532" "False"