#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import zipfile
import numpy as np
import pandas as pd
import skimage.io as skio

if __name__ == '__main__':
    directory = '/mnt/82db778e-0496-450c-9b25-d1e50a90e476/data/remote-sensing-image-segmentation/02_test_clean/'
    fidx = directory + 'idx-test.txt'
    wdir = os.path.dirname(fidx)
    dataCSV = pd.read_csv(fidx)
    lst_cls = [0, 127, 255]
    lst_idx = []
    lst_lbl = None
    for ii, (nrow, ncol, fimg) in enumerate(zip(dataCSV['out_rows'], dataCSV['out_cols'], dataCSV['path_img'])):
        fimg = os.path.join(wdir, fimg)
        msk_fake = np.zeros((nrow, ncol), dtype=np.uint8)
        numcD3 = msk_fake.shape[1] // 3
        #
        for kki in range(3):
            msk_fake[:, (kki * numcD3): ((kki + 1) * numcD3)] = lst_cls[kki]
        fimgFake = fimg.replace('.png', '_msk_fake.png')
        skio.imsave(fimgFake, msk_fake)
        fID = os.path.splitext(os.path.basename(fimg))[0]
        tmskF = msk_fake.flatten()
        lst_idx += ['{}_{}'.format(fID, xx) for xx in range(tmskF.shape[0])]
        if lst_lbl is None:
            lst_lbl = tmskF
        else:
            lst_lbl = np.concatenate((lst_lbl, tmskF))
        print('[{}/{}]'.format(ii, len(dataCSV)))
    csvData = pd.DataFrame(data={
        'idx': lst_idx,
        'msk': lst_lbl
    }, columns=['idx', 'msk'])
    foutCSV = '{}_predict_fake.csv'.format(fidx)
    csvData.to_csv(foutCSV, sep=',', index=None)
    foutCSV_Zip = '{}.zip'.format(foutCSV)
    with zipfile.ZipFile(foutCSV_Zip, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(foutCSV, arcname=os.path.basename(foutCSV))
