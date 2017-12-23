#!/bin/bash

script_location='/scratch/aolpin/thesis/code/log.py'
output_dir='/scratch/aolpin/thesis/results/'
datset_dir='/scratch/aolpin/thesis/dataset/'
log_dir='/scratch/aolpin/thesis/logs/'
randoms='./randomsip.log'

echo "Read file"
while IFS= read -r line
do
  	#echo "$line"
  	IFS=',' read -ra var <<< "$line"
    date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
	sqsub -r 4h -o "${log_dir}log_$date_time.log" python $script_location ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} $datset_dir
	
done < "$randoms"
