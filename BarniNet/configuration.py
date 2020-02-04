
"""
    2017-2018 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:
    M. Barni, A. Costanzo, E. Nowroozi, B. Tondi., “CNN-based detection of generic contrast adjustment with
    JPEG post-processing", ICIP 2018 (http://clem.dii.unisi.it/~vipp/files/publications/CM-icip18.pdf)

"""

# ---------------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------------

DATASET_FOLDER = '...........................'         # Dataset directory

PATCH_SIZE = 64                                 # Patch size
PATCH_CHANNELS = 1                              # Color mode

CLASS_0_TAG = 'pristine'                        # Tag for pristine image class  LABEL 0
CLASS_1_TAG = 'median05'                        # Tag for enhanced image class   LABEL 1

#MAX_TRAIN_BLOCKS = 2e6                          # Number of training patches
#MAX_VAL_BLOCKS = 2e5                            # Number of validation patches
#MAX_TEST_BLOCKS = 2e5                           # Number of test patches

MAX_TRAIN_BLOCKS = 500000                        # Number of training patches
MAX_VAL_BLOCKS = 5000                           # Number of validation patches
MAX_TEST_BLOCKS = 10000                           # Number of test patches
MAX_PER_IMAGE = 100                            # Maximum number of patches from the same image

# ---------------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------------

MODEL_FOLDER = './models/'                      # Models are stored here
TRAIN_FOLDER = r'...........................'                     # Training patches are stored here
VALIDATION_FOLDER =r'...........................'           # Validation patches are stored here
TEST_FOLDER = r'...........................'                      # Test patches are stored here

# ---------------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------------

NUM_EPOCHS = 4                                  # Number of training epochs
TRAIN_BATCH = 64                                  # Training batch size
VALIDATION_BATCH = 64                         # Validation batch size
TEST_BATCH = 100                                # Test batch size

# Resume training

RESUME_TRAINING = False
RESUME_MODEL = '.................................h5'

# Augmentations

JPEG_AUGMENTATION = False
AUG_JPEG_QFS = [ -1]
