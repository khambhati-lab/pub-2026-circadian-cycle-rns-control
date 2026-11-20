import sys
import os
import multiprocessing
import datetime
import logging

import numpy as np
import pandas as pd


import src.dataproc.neuropaceIO as neuropaceIO
from src.dataproc import stimartifact
from src.dataproc import edgecases
# import pdb

from omegaconf import OmegaConf
# import logging

logging.basicConfig(filename=f"logs/log_{datetime.datetime.now()}.txt", level=logging.DEBUG)
logging.captureWarnings(True)
logging.info('Running from src/runexps/stimcounter')


def count_stims_in_ecogs(folder_path, df_catalog_pt, params):

    scountdf = df_catalog_pt.copy()
    scountdf['stim_count'] = np.NaN
    scountdf['num_timepoints_retained'] = np.NaN
    scountdf['frac_timepoints_retained'] = np.NaN
    
    # for epinum in range(df_catalog_pt.shape[0]): # each episode separately
    for epiidx, row in df_catalog_pt.iterrows(): # each episode separately
        # get data from an episode
        if epiidx % 1000 == 0:
            print(f'Extracting features from episode: {epiidx}')

        data_dict = neuropaceIO.get_ieeg_data_dict(df_catalog_pt.loc[[epiidx]], folder_path)
        
        ptsmat = data_dict['signal'].copy() # take the entire signal
        num_T = ptsmat.shape[0]
        keep_time_indices = np.arange(num_T, dtype=float) # indices of all the times to keep
        
        if num_T > 0:
            # times that contain stimulation artifacts
            stimart_times_list, stim_times_list, ignore2 = stimartifact.detect_stim_artifacts(signalmat=ptsmat, params=params.stim)

            # time points to remove in filtered signal
            if len(stimart_times_list) > 0: # only do for loop if stimulation artifacts were detected in the data
                for ii in range(stimart_times_list.shape[0]):
                    # clip 1s before stim artifact begins or till start of signal, whichever is larger
                    t_clip_start = max(0, stimart_times_list[ii][0] - params.stim.SIDE_CLIP_LEN)
                    # clip 1s after stim artifact ends or till max len of signal, whichever is smaller
                    t_clip_end = min(stimart_times_list[ii][1] + params.stim.SIDE_CLIP_LEN, num_T)
        
                    keep_time_indices[t_clip_start:t_clip_end] = np.NaN

            # drop the indices that are NaN
            keep_time_indices = keep_time_indices[~np.isnan(keep_time_indices)]
            keep_time_indices = keep_time_indices.astype(int)  # convert to int to allow indexing
    
            # save stim counts
            scountdf.loc[epiidx, 'stim_count'] = len(stimart_times_list)
            scountdf.loc[epiidx, 'num_timepoints_retained'] = len(keep_time_indices)
            scountdf.loc[epiidx, 'frac_timepoints_retained'] = len(keep_time_indices) / num_T
            
        else:
            scountdf.loc[epiidx, 'stim_count'] = 0
            scountdf.loc[epiidx, 'num_timepoints_retained'] = num_T
            scountdf.loc[epiidx, 'frac_timepoints_retained'] = 1
            
    return scountdf



def extract_ECoG_feature_wrapper(patient_code, pathsdict, paramsdict):

    # load dataframes of catalog files for given NPE code
    if os.path.exists(pathsdict.df_catalog_path):
        df_catalog = pd.read_csv(pathsdict.df_catalog_path) # catalog file
    else:
        logging.debug(f'Catalog file not found: {pathsdict.df_catalog_path}.')
        print(f'Catalog file not found: {pathsdict.df_catalog_path}.')
        return

    # get subset of catalog related to given trigger type (eg Scheduled, Long_Episode)
    df_catalog_pt = df_catalog.loc[(df_catalog['ECoG trigger'].isin(['Scheduled', 'Long_Episode']) )].copy()

    # handle any edge cases in catalog file
    df_catalog_pt = edgecases.catalogdf_edgecases(patient_code, df_catalog_pt)

    # compute feature
    scountdf = count_stims_in_ecogs(folder_path=pathsdict.ecog_folder_path, df_catalog_pt=df_catalog_pt,
                                           params=paramsdict)

    # save the dataframe
    scountdf.to_csv(pathsdict.save_feature_path, index=False)


def main_featext(input_dict):

    # parse input from input dictionary
    machine, patient_code = input_dict['machine'], input_dict['npcode']
    print(f'In cluster implementation for {patient_code}')

    ##### parameters
    params = OmegaConf.load('src/conf/features/stimcounter.yaml')

    ##### paths
    paths = OmegaConf.load(f'src/conf/paths/{machine}_stim.yaml')
    paths.df_catalog_path = paths.df_catalog_path % (patient_code, patient_code)
    paths.ecog_folder_path = paths.ecog_folder_path % (patient_code, patient_code)
    paths.save_feature_path = paths.save_feature_path % (patient_code)

    ##### compute feature
    extract_ECoG_feature_wrapper(patient_code=patient_code, pathsdict=paths,  paramsdict=params)



if __name__ == "__main__":

    print('In main.')

    machine = 'cluster' # 'local' or 'cluster'
    patient_list = ['NPE%04d'%(ii+1) for ii in range(138)]

    args_list = [{'machine':machine, 'npcode':patient_id} for patient_id in patient_list]

    pool = multiprocessing.Pool(processes=20)
    recresult_list = pool.map(main_featext, args_list)
    
    pool.close()
    print('Pool computation started')
    pool.join()
    print('Pool computation ended')

    
