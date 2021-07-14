#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import tensorflow as tf
from miscnn.data_loading.interfaces import NIFTI_interface, DICOM_interface
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, \
                                          dice_crossentropy, tversky_loss
import argparse
import os

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
def ggo_prediction(img_path, uid, date):


    path_model = './ggo_seg/scripts/models/model.hdf5'
    path_preds = '/home/ubuntu/uploads/'+uid+'/'+date+'/nifti'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    print('output of ggo at ',path_preds)
    #-----------------------------------------------------#
    #               Setup of MIScnn Pipeline              #
    #-----------------------------------------------------#
    # Initialize Data IO Interface for NIfTI data
    ## We are using 4 classes due to [background, lung_left, lung_right, covid-19]
    interface = NIFTI_interface(channels=1, classes=4)

    # Create Data IO object to load and write samples in the file structure
    data_io = Data_IO(interface, input_path=img_path, output_path=path_preds,
                    delete_batchDir=False)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    # Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
    sf_clipping = Clipping(min=-1250, max=250)
    # Create a pixel value normalization Subfunction to scale between 0-255
    sf_normalize = Normalization(mode="grayscale")
    # Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
    sf_resample = Resampling((1.58, 1.58, 2.70))
    # Create a pixel value normalization Subfunction for z-score scaling
    sf_zscore = Normalization(mode="z-score")

    # Assemble Subfunction classes into a list
    sf = [sf_clipping, sf_normalize, sf_resample, sf_zscore]

    # Create and configure the Preprocessor class
    pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                    prepare_subfunctions=True, prepare_batches=False,
                    analysis="patchwise-crop", patch_shape=(160, 160, 80),
                    use_multiprocessing=True)
    # Adjust the patch overlap for predictions
    pp.patchwise_overlap = (80, 80, 30)
    pp.mp_threads = 16

    # Initialize the Architecture
    unet_standard = Architecture(depth=4, activation="softmax",
                                batch_normalization=True)

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, architecture=unet_standard,
                        loss=tversky_crossentropy,
                        metrics=[tversky_loss, dice_soft, dice_crossentropy],
                        batch_queue_size=4, workers=4, learninig_rate=0.001)

    # Load best model weights during fitting
    # model.load(path_model)

    # Compute predictions
    model.predict_trt(sample_list, return_output=False)
