import SimpleITK as sitk
import os
print('Start')
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
print(file)
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/PRED.nii.gz'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse <= 10:
    print('Test passed!')
else:
    print(f'Test failed! MSE={mse}')