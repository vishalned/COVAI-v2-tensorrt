import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from glob import glob
from lungmask import mask
from lungmask import utils


force_cpu=False
batch_size=4
volume_postprocessing=True
noHU=False

# input_folder = './content/input_files/'
# output_folder = './predictions'


def numpy2nifti(input_file, seg_file, output_folder):
  func = nib.load(input_file)
  input_name = input_file.split('/')[-1].split('.')[-3]
  ni_img = nib.Nifti1Image(seg_file, func.affine)
  nib.save(ni_img, f'{output_folder}processed_lobe.nii.gz')

  
def lobe_prediction(img, uid, date):
  output_folder = '../uploads/'+uid+'/'+date+'/nifti/'
  image = sitk.ReadImage(img)
  mdl_r = mask.get_model('unet', 'R231')
  mdl_l = mask.get_model('unet', 'LTRCLobes')
  res_l = mask.apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
  res_r = mask.apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
  spare_value = res_l.max()+1
  res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
  res_l[res_r==0] = 0

  total_slices = res_l.shape[0]
  no_silces = total_slices // 5

  # 5 parts model
  seg1 = res_l[:no_silces, :, :]
  seg2 = res_l[no_silces:2*no_silces, :, :]
  seg3 = res_l[2*no_silces:3*no_silces, :, :]
  seg4 = res_l[3*no_silces:4*no_silces, :, :]
  seg5 = res_l[4*no_silces:, :, :]

  pred1 = utils.postrocessing(seg1, spare=[spare_value])
  pred2 = utils.postrocessing(seg2, spare=[spare_value])
  pred3 = utils.postrocessing(seg3, spare=[spare_value])
  pred4 = utils.postrocessing(seg4, spare=[spare_value])
  pred5 = utils.postrocessing(seg5, spare=[spare_value])

  ensemble = np.vstack((pred1, pred2, pred3, pred4, pred5))

  output = np.rollaxis(ensemble, 0, 3)
  output = np.rot90(output)
  output = np.flip(output, axis=0)
  output = output.astype(np.uint16)
  numpy2nifti(img, output, output_folder)

        
    
    
