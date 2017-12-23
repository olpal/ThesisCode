#!/bin/bash

script_location='/scratch/aolpin/thesis/code/training.py'
script_locationip='/scratch/aolpin/thesis/code/log.py'
output_dir='/scratch/aolpin/thesis/results/'
datset_dir='/scratch/aolpin/thesis/dataset/'
datset_dirip='/scratch/aolpin/execdata/'
log_dir='/scratch/aolpin/thesis/logs/'
randomsip='./randomsip.log'
randoms='./randoms.log'
randomsf='./randomsf.log'

echo "Read file"
while IFS= read -r line
do
  	#echo "$line"
  	IFS=',' read -ra var <<< "$line"
    date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
	  sqsub -r 4h -q gpu -f threaded -n 4 --gpp=8 --mpp=24g -o "${log_dir}cnn-${var[7]}-${var[8]}-$date_time.log" python $script_location $datset_dir "${output_dir}cnn/${date_time}" ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} ${var[6]} "true" ${var[7]} ${var[8]}

done < "$randoms"

echo "Read file"
while IFS= read -r line
do
  	#echo "$line"
  	IFS=',' read -ra var <<< "$line"
    date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
	  sqsub -r 4h -q gpu -f threaded -n 4 --gpp=8 --mpp=24g -o "${log_dir}fcn-${var[7]}-${var[8]}-$date_time.log" python $script_location $datset_dir "${output_dir}fcn/${date_time}" ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} ${var[6]} "false" ${var[7]} ${var[8]}

done < "$randomsf"

echo "Read file"
while IFS= read -r line
do
  	#echo "$line"
  	IFS=',' read -ra var <<< "$line"
    date_time=`date +%Y-%m-%d--%H-%M-%S-%N`
	  sqsub -r 60m -o "${log_dir}log_$date_time.log" python $script_locationip ${var[0]} ${var[1]} ${var[2]} ${var[3]} ${var[4]} ${var[5]} $datset_dirip "${output_dir}log/${date_time}"

done < "$randomsip"