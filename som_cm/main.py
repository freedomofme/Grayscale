# -*- coding: utf-8 -*-
## @package som_cm.main
#
#  Main functions.
#  @author      tody
#  @date        2015/08/19
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.google_image import createDatasets
from results.single_image import singleImageResults
from results.multi_images import multiImagesResults

if __name__ == '__main__':

    data_names = ["apple"]
    num_images = 5

    # createDatasets(data_names, num_images, update=False)
    data_ids = range(num_images)
    singleImageResults(data_names, data_ids)

    # data_ids = range(7)
    # multiImagesResults(data_names, data_ids)
