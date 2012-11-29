#!/usr/bin/env bash

# check if running locally or on DAS-4
if [ $1 = local ]
then
    COMMAND="./assign3_1"
else
    COMMAND="prun -v -n1 assign3_1"
fi

# setup the OMP environment variables
export OMP_SCHEDULE="STATIC"

N=4
I_VALS=(1000 10000 1000000 10000000)
T_VALS=(1000000 500000 5000 500)

# clear the output file
echo > exp_results.txt

for ((i=0; i < 4; i++))
do
    # loop through the number of threads
    for threads in 1 2 4 8 16; do
        export OMP_NUM_THREADS=$threads
        echo Doing ${I_VALS[$i]} ${T_VALS[$i]} with $OMP_NUM_THREADS threads

        echo Parameters: ${I_VALS[i]} ${T_VALS[$i]} $OMP_NUM_THREADS >> exp_results.txt
        $COMMAND ${I_VALS[$i]} ${T_VALS[$i]} 1 >> exp_results.txt
        echo -e "\n\n" >> exp_results.txt
    done
done
