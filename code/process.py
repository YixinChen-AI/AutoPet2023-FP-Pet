import SimpleITK
import numpy as np
import torch
from STUNet import STUNet
from Monai_model import Monai_model
import os
import shutil
from utils import *
import nibabel as nib

class Unet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/'  # where to store the nii files
        # self.ckpt_path0 = '/opt/algorithm/STUNet_base_ctpet_fold0.pth'
        self.ckpt_path0 = '/opt/algorithm/v3_STUNet_large_ctpet_fold-1_punlum_tversky_softdice3d.pth'
        # self.ckpt_path3 = '/opt/algorithm/STUNet_base_ctpet_fold3_punlum.pth'
        
#         # for test
        # self.input_path = './test/input/'  # according to the specified grand-challenge interfaces
        # self.output_path = './test/output/'  # according to the specified grand-challenge interfaces
        # self.nii_path = './test/output/'  # where to store the nii files
        # # self.ckpt_path0 = './STUNet_base_ctpet_fold0.pth'
        # self.ckpt_path0 = './v2_STUNet_large_ctpet_fold-1_punlum_tversky_softdice3d.pth'
        # # self.ckpt_path3 = './STUNet_base_ctpet_fold3_punlum.pth'

        self.model0 = create_model("STUNet_large")
        self.model0 = self.model0.to("cuda:0")
        self.model0.load_state_dict(torch.load(self.ckpt_path0))
        self.model0._deep_supervision = False
        self.model0.eval()
        # self.model1 = create_model("STUNet_base")
        # self.model1 = self.model1.to("cuda:0")
        # self.model1.load_state_dict(torch.load(self.ckpt_path1))
        # self.model1._deep_supervision = False
        # self.model1.eval()
        # self.model3 = create_model("STUNet_base")
        # self.model3 = self.model3.to("cuda:0")
        # self.model3.load_state_dict(torch.load(self.ckpt_path3))
        # self.model3._deep_supervision = False
        # self.model3.eval()
#         self.model4 = create_model("STUNet_base")
#         self.model4 = self.model4.to("cuda:0")
#         self.model4.load_state_dict(torch.load(self.ckpt_path4))
#         self.model4._deep_supervision = False
#         self.model4.eval()

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self,ct_mha,pet_mha):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        uuid = os.path.splitext(ct_mha)[0]
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"), os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))
    
    def predict(self):
        """
        Your algorithm goes here
        """  
        inference(self.model0,self.nii_path,self.output_path,
                 model1=None,
                 model2=None,
                 model3= None,
                 model4 = None)
    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        ct_mha = sorted(os.listdir(os.path.join(self.input_path, 'images/ct/')))
        ct_mha = [i for i in ct_mha if "mha" in i]
        pet_mha = sorted(os.listdir(os.path.join(self.input_path, 'images/pet/')))
        pet_mha = [i for i in pet_mha if "mha" in i]
        print(ct_mha,pet_mha)
        for ct,pet in zip(ct_mha,pet_mha):
            
            uuid = self.load_inputs(ct,pet)
            print("uuid",uuid)
            print('Start prediction')
            self.predict()
            print('Start output writing')
            self.write_outputs(uuid)
#         break

import monai.transforms as transforms
import torch,monai


def inference(model,path,output_path,model1=None,model2=None,model3=None,model4=None,):
    ct_path = os.path.join(path,"CTres.nii.gz")
    pet_path = os.path.join(path,"SUV.nii.gz")

    pet_itk = sitk.ReadImage(pet_path)
    ct_itk = sitk.ReadImage(ct_path)

    spacing = pet_itk.GetSpacing()
#     pet_itk = resampleVolume((2,2,3),pet_itk,resamplemethod=sitk.sitkLinear)
#     ct_itk = resampleVolume((2,2,3),ct_itk,resamplemethod=sitk.sitkLinear)

    pet = sitk.GetArrayFromImage(pet_itk)
    ct = sitk.GetArrayFromImage(ct_itk)

    pet = pet.copy()
    ct = ct.copy()

    pet = torch.tensor(pet).float()
    ct = torch.tensor(ct).float()

    normal_transforms = transforms.Compose([
            transforms.ScaleIntensityRanged(
                keys=["image_ct"], a_min=-1000, a_max=1000,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            transforms.ScaleIntensityRanged(
                keys=["image_pt"], a_min=0, a_max=30,
                b_min=0.0, b_max=1.0, clip=True,
            ),                            
            transforms.ConcatItemsd(keys=["image_pt", "image_ct"], name="image_petct", dim=0),# concatenate pet and ct channels
        ])

    pet = pet.float().unsqueeze(0)
    ct = ct.float().unsqueeze(0)

    datadict = normal_transforms({"image_ct":ct,"image_pt":pet})
    x_ori = datadict["image_petct"]
    x_ori = x_ori.to("cuda:0").unsqueeze(0)
    
    transforms1 = transforms.RandFlipd(keys=["image_petct"], prob=1, spatial_axis=0)
    transforms2 = transforms.RandFlipd(keys=["image_petct"], prob=1, spatial_axis=1)
    transforms3 = transforms.RandFlipd(keys=["image_petct"], prob=1, spatial_axis=2)
    
    with torch.no_grad():
        pet,ct,crop_info = adaptive_center_crop(x_ori[0,0],x_ori[0,1])
        x_crop = torch.cat([pet[None],ct[None]],dim=0).unsqueeze(0)
        
        wb_pred = monai.inferers.sliding_window_inference(
                        x_crop,roi_size=(128,128,128),sw_batch_size=2,
                        predictor=model,overlap=0.75,mode="gaussian",
                        sw_device="cuda:0",device="cpu",progress=False)
        wb_pred = torch.softmax(wb_pred,dim=1)
        
        
        # flip dim=2
        datadict = transforms1({"image_petct":x_crop[0]})
        x = datadict["image_petct"]
        x = x.unsqueeze(0)
        wb_pred1 = monai.inferers.sliding_window_inference(
            x,roi_size=(128,128,128),sw_batch_size=2,
            predictor=model,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
        wb_pred1 = torch.flip(wb_pred1,[2])
        wb_pred += torch.softmax(wb_pred1,dim=1)
        
        # flip dim=3
        datadict = transforms2({"image_petct":x_crop[0]})
        x = datadict["image_petct"]
        x = x.unsqueeze(0)
        wb_pred2 = monai.inferers.sliding_window_inference(
            x,roi_size=(128,128,128),sw_batch_size=2,
            predictor=model,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
        wb_pred2 = torch.flip(wb_pred2,[3])
        wb_pred += torch.softmax(wb_pred2,dim=1)
        
        
        # flip dim=4
        datadict = transforms3({"image_petct":x_crop[0]})
        x = datadict["image_petct"]
        x = x.unsqueeze(0)
        wb_pred3 = monai.inferers.sliding_window_inference(
            x,roi_size=(128,128,128),sw_batch_size=2,
            predictor=model,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
        wb_pred3 = torch.flip(wb_pred3,[4])
        wb_pred += torch.softmax(wb_pred3,dim=1)

        other_models = []
        if model1 is not None:
            other_models.append(model1)
        if model2 is not None:
            other_models.append(model2)
        for model_ in other_models:
            wb_pred01 = monai.inferers.sliding_window_inference(
                        x_crop,roi_size=(128,128,128),sw_batch_size=2,
                        predictor=model_,overlap=0.75,mode="gaussian",
                        sw_device="cuda:0",device="cpu",progress=False)
            wb_pred01 = torch.softmax(wb_pred01,dim=1)
            wb_pred = torch.softmax(wb_pred01,dim=1)


            # flip dim=2
            datadict = transforms1({"image_petct":x_crop[0]})
            x = datadict["image_petct"]
            x = x.unsqueeze(0)
            wb_pred02 = monai.inferers.sliding_window_inference(
                x,roi_size=(128,128,128),sw_batch_size=2,
                predictor=model_,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
            wb_pred02 = torch.flip(wb_pred02,[2])
            wb_pred += torch.softmax(wb_pred02,dim=1)

            # flip dim=3
            datadict = transforms2({"image_petct":x_crop[0]})
            x = datadict["image_petct"]
            x = x.unsqueeze(0)
            wb_pred03 = monai.inferers.sliding_window_inference(
                x,roi_size=(128,128,128),sw_batch_size=2,
                predictor=model_,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
            wb_pred03 = torch.flip(wb_pred03,[3])
            wb_pred += torch.softmax(wb_pred03,dim=1)


            # flip dim=4
            datadict = transforms3({"image_petct":x_crop[0]})
            x = datadict["image_petct"]
            x = x.unsqueeze(0)
            wb_pred04 = monai.inferers.sliding_window_inference(
                x,roi_size=(128,128,128),sw_batch_size=2,
                predictor=model_,overlap=0.75,mode="gaussian",sw_device="cuda:0",device="cpu",progress=False)
            wb_pred04 = torch.flip(wb_pred04,[4])
            wb_pred += torch.softmax(wb_pred04,dim=1)
        
        
        # padding to original size
        wb_pred = torch.argmax(wb_pred,dim=1)

        wb_pred_ = torch.zeros_like(x_ori[0,0])
        wb_pred_[:,crop_info[0]:-crop_info[2],crop_info[1]:-crop_info[3]] = wb_pred.cpu()
        wb_pred = wb_pred_
    
#     wb_pred = torch.argmax(wb_pred,dim=1)
    wb_pred = wb_pred.detach().cpu().numpy()
    wb_pred = wb_pred.astype(np.uint8)   
    
#     PT = nib.load(os.path.join(path,"SUV.nii.gz"))  #needs to be loaded to recover nifti header and export mask
#     pet_affine = PT.affine
#     PT = PT.get_fdata()
#     mask_export = nib.Nifti1Image(wb_pred, pet_affine)
#     print(os.path.join(output_path, "PRED.nii.gz"))

#     nib.save(mask_export, os.path.join(output_path,"PRED.nii.gz"))
    print(wb_pred.shape)
    out = sitk.GetImageFromArray(wb_pred)
#     out.SetSpacing((2,2,3))
#     out = resampleVolume(spacing,ct_itk,resamplemethod=sitk.sitkNearestNeighbor)
    sitk.WriteImage(out, os.path.join(output_path,"PRED.nii.gz"))

def create_model(modelname):
    model = None
    num_input_channels = 2
    num_classes = 2
    if modelname == "STUNet_small":
        model = STUNet(num_input_channels, 105, depth=[1,1,1,1,1,1], dims=[16, 32, 64, 128, 256, 256],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
        model.seg_outputs[0] = nn.Conv3d(256,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[1] = nn.Conv3d(128,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[2] = nn.Conv3d(64,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[3] = nn.Conv3d(32,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[4] = nn.Conv3d(16,num_classes,kernel_size=1,stride=1)
        
    if modelname == "STUNet_base":
        model = STUNet(1, 105, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
        model.seg_outputs[0] = nn.Conv3d(512,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[1] = nn.Conv3d(256,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[2] = nn.Conv3d(128,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[3] = nn.Conv3d(64,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[4] = nn.Conv3d(32,num_classes,kernel_size=1,stride=1)
        model.conv_blocks_context[0][0].conv1 = nn.Conv3d(num_input_channels,32,kernel_size=(3,3,3),stride=1,padding=1)
        model.conv_blocks_context[0][0].conv3 = nn.Conv3d(num_input_channels,32,kernel_size=(1,1,1),stride=1,padding=0)        
        
    if modelname == "STUNet_large":
        model = STUNet(1, 105, depth=[2,2,2,2,2,2], dims=[64, 128, 256, 512, 1024, 1024],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
        model.seg_outputs[0] = nn.Conv3d(1024,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[1] = nn.Conv3d(512,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[2] = nn.Conv3d(256,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[3] = nn.Conv3d(128,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[4] = nn.Conv3d(64,num_classes,kernel_size=1,stride=1)
        model.conv_blocks_context[0][0].conv1 = nn.Conv3d(num_input_channels,64,kernel_size=(3,3,3),stride=1,padding=1)
        model.conv_blocks_context[0][0].conv3 = nn.Conv3d(num_input_channels,64,kernel_size=(1,1,1),stride=1,padding=0)    
    if modelname == "STUNet_huge":
        model = STUNet(1,105, depth=[3,3,3,3,3,3], dims=[96, 192, 384, 768, 1536, 1536],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
        model.seg_outputs[0] = nn.Conv3d(1536,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[1] = nn.Conv3d(768,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[2] = nn.Conv3d(384,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[3] = nn.Conv3d(192,num_classes,kernel_size=1,stride=1)
        model.seg_outputs[4] = nn.Conv3d(96,num_classes,kernel_size=1,stride=1)
        model.conv_blocks_context[0][0].conv1 = nn.Conv3d(num_input_channels,96,kernel_size=(3,3,3),stride=1,padding=1)
        model.conv_blocks_context[0][0].conv3 = nn.Conv3d(num_input_channels,96,kernel_size=(1,1,1),stride=1,padding=0)    
    
    if modelname =="SwinUNETR":
        model = monai.networks.nets.SwinUNETR(
                img_size=input_size,
                in_channels=num_input_channels,
                out_channels=num_classes,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                dropout_path_rate=0.1,
            )
    if modelname == "VNet":
        model = monai.networks.nets.VNet(
                spatial_dims=3,
                in_channels=num_input_channels,
                out_channels=num_classes,
            )
    if modelname == "UNet":
        model = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=num_input_channels,
                out_channels=num_classes,
                channels=(32,64,128,256,512),
                strides=(2,2,2,2),
                num_res_units=2,
                norm=monai.networks.layers.Norm.BATCH,
            )
    assert model != None
    return model 
    
def adaptive_center_crop(img,ct):
    row = torch.mean(ct,dim=[0,2])
    col = torch.mean(ct,dim=[0,1])
    
    start = 0
    start1 = 0
    start2 = 0
    start3 = 0
    
    for i in range(5):
        if row[start+20] ==0:
            start = start + 20
        
        if row[-start2-20] ==0:
            start2 = start2 + 20
            
        if col[start1+20] ==0:
            start1 = start1 + 20
            
        if col[-start3-20] ==0:
            start3 = start3 + 20
    
        
    img = img[:,start:-start2,start1:-start3]
    ct = ct[:,start:-start2,start1:-start3]    
    return img,ct,[start,start1,start2,start3]
# def adaptive_center_crop(img,ct):
#     row = torch.mean(ct,dim=[0,2])
#     col = torch.mean(ct,dim=[0,1])
    
#     start = 0
#     start1 = 0
    
#     for i in range(5):
#         if row[start+20] == 0 and row[-start-20] == 0:
#             start = start + 20
            
#         if col[start1+20] == 0 and col[-start1-20] == 0:
#             start1 = start1 + 20
        
#     img = img[:,start:-start,start1:-start1]
#     ct = ct[:,start:-start,start1:-start1]    
#     return img,ct
if __name__ == "__main__":
    Unet_baseline().process()
