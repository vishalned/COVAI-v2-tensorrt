import nibabel as nib
import numpy as np
from ggo_seg.scripts.predict import *
from lung_lobe import *
import time
import threading
import dicom2nifti
import sys
from numba import cuda
import tensorflow as tf
from tensorflow.python.keras import backend as K




config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
Session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(Session)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

uid = sys.argv[1]
date = sys.argv[2]
#_, uid, date = str(sys.argv)

input_folder = '../uploads/'+uid+'/'+date+'/scans/'
output_folder = '../uploads/'+uid+'/'+date+'/nifti/'
# exists = os.path.isdir(output_folder)
# if not exists:
#     os.mkdir(output_folder)
# try:
#     print("processing dcm to nii")
#     dicom2nifti.convert_directory(input_folder, output_folder)
# except:
#     pass
# os.rename((output_folder + os.listdir(output_folder)[0]) , output_folder + "unprocessed.nii.gz")

IMG_DIR = output_folder
IMG_PATH = output_folder + 'unprocessed.nii.gz'

start = time.time()



start_ggo = time.time()
#ggo_prediction(IMG_DIR, uid, date)
t1 = threading.Thread(target = ggo_prediction, kwargs = {'img_path' : IMG_DIR, 'uid' : uid, 'date' : date})
t1.start()
end_ggo = time.time()
print("ggo prediction complete in ", (end_ggo - start_ggo), " seconds")

#device = cuda.get_current_device()
#device.reset()


start_lobe = time.time()
#lobe_prediction(IMG_PATH, uid, date)
t2 = threading.Thread(target = lobe_prediction, kwargs = {'img' : IMG_PATH, 'uid' : uid, 'date' : date})
t2.start()
end_lobe = time.time()
print("lobe segmentation complete in ",(end_lobe - start_lobe), " seconds")

t1.join()
t2.join()
print("starting post processing")
start_pp = time.time()
img_ggo  = nib.load(output_folder + 'processed_ggo.nii.gz')
img_lobe = nib.load(output_folder + 'processed_lobe.nii.gz')
data_ggo = img_ggo.get_fdata()
data_lobe = img_lobe.get_fdata()

# _, counts_data_ggo = np.unique(data_ggo, return_counts = True)
# _, counts_data_lobe = np.unique(data_lobe, return_counts = True)

# total_lung_count = counts_data_ggo[1] + counts_data_ggo[2] + counts_data_ggo[3]
# total_ggo_count = counts_data_ggo[3]

ggo_lobe = {"lobe "+str(i):np.count_nonzero(data_ggo[data_lobe == (i+1)] == 3) for i in range(5)}
lobe_area = {"lobe "+str(i):np.count_nonzero(data_lobe == (i+1)) for i in range(5)}

# print(ggo_lobe)
# print(lobe_area)

percentages = np.array(list(ggo_lobe.values())) / np.array(list(lobe_area.values())) * 100
# print(percentages)
# print(percentages)
percentages_dict = {}
percentages_dict['lu'] = percentages[0]
percentages_dict['ll'] = percentages[1]
percentages_dict['ru'] = percentages[2]
percentages_dict['rm'] = percentages[3]
percentages_dict['rl'] = percentages[4]

end_pp = time.time()
print(percentages_dict)

print("post processing completed in ",(end_pp - start_pp), " seconds")

end = time.time()
print("full model completed in ",(end-start), " seconds")



# 0 -  outside the lungs
# 1 -  left lung, upper lobe
# 2 -  left lung, lower lobe
# 3 -  right lung, upper lobe
# 4 -  right lung, middle lobe
# 5 -  right lung, lower lobe.


