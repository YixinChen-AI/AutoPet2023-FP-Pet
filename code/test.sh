#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create unet_baseline-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
# Do not change any of the parameters to docker run, these are fixed

docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --gpus="all"  \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v unet_baseline-output-$VOLUME_SUFFIX:/output/ \
        unet_baseline

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t unet_eval .

docker run --rm -it \
        -v unet_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output/:/expected_output/ \
        unet_eval python3 -c """
        
        
import SimpleITK as sitk
import os
import numpy as np
print('Start')
files = os.listdir('/output/images/automated-petct-lesion-segmentation')
print(files)
files = [f for f in files if '.mha' in f]
for file in files:
    output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
    expected_output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/expected_output/',file)))
    print(np.sum(output),np.max(output))
    mse = sum(sum(sum((output - expected_output) ** 2)))
    if mse <= 10:
        print('Test passed!')
    else:
        print(f'Test failed! MSE={mse}')
    break
"""



docker volume rm unet_baseline-output-$VOLUME_SUFFIX
