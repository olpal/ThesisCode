#!/bin/bash

output_dir='/Users/aolpin/Documents/School/thesis/results/testresults/'
datset_dir='/Users/aolpin/Documents/School/thesis/dataset/'
log_dir='/scratch/aolpin/thesis/logs/'
randoms='./randoms.log'

echo "Read file"
while IFS= read -r line
do
  	#echo "$line"
  	IFS=',' read -ra var <<< "$line"
    date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
	#python2 ./training.py $datset_dir "${output_dir}cnn/${date_time}" ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} ${var[6]} "True" ${var[7]}
	python2 ./training.py $datset_dir "${output_dir}fcn/${date_time}" ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} ${var[6]} "False" ${var[7]}

done < "$randoms"