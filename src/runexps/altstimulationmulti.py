import multiprocessing

import pandas as pd
import numpy as np
import pickle

from src.model import kuramotoattractor
from src.dataproc import ccutils

'''
Main routine to execute several alternate stimulation tasks in parallel.

Date, Author: --, Krishnakant V. Saboo
'''


def coupling_prediction_models():
    #### paths
    klrmodel_path = 'data/processed/kuramotoattractor/Klr_pred_models.pkl'
    kstimmodel_path = 'data/processed/kuramotoattractor/Kstim_pred_models.pkl'
    aucintmodel_path = 'data/processed/kuramotoattractor/hour_auc_interval_map.pkl'
    aucconmodel_path = 'data/processed/kuramotoattractor/hour_auc_converse_map.pkl'

    ##### coupling and stimulation AUC prediction models
    klrmodel = pickle.load(open(klrmodel_path, 'rb'))
    kstimmodel = pickle.load(open(kstimmodel_path, 'rb'))
    aucintmodel = pickle.load(open(aucintmodel_path, 'rb'))
    aucconmodel = pickle.load(open(aucconmodel_path, 'rb'))

    return klrmodel, kstimmodel, aucintmodel, aucconmodel



def cycle_info_dataframe():

    #### paths
    fitted_coupling_filepath = 'data/processed/kuramotoattractor/fitted_couplings_attractor_brain_stim_24Sep2025.csv'
    
    #### features of FC-CC and stimulation
    regdf = pd.read_csv(fitted_coupling_filepath)  # use pre-saved dataframe

    # sin and cos of peak times in regdf
    for col in ['left_peakphase_prev','right_peakphase_prev', 'left_peakphase', 'right_peakphase', 'stim_peakphase']:
        if 'prev' in col:
            colsuffix = col.split('_')[0] + '_prev'
        else:
            colsuffix = col.split('_')[0]
            
        regdf[f'sin_{colsuffix}'] = np.sin(regdf[col])
        regdf[f'cos_{colsuffix}'] = np.cos(regdf[col])

    # summary features of FC-CC cycles
    regdf['left_right_peakgap_prev'] = regdf.apply(lambda x: ccutils.angle_gap(x, 'left_peaktime_prev', 'right_peaktime_prev'), 
                                      axis=1)
    regdf['total_amp_prev'] = regdf['left_amp_prev'] + regdf['right_amp_prev']

    return regdf




def alt_stim_for_one_index(input_dict):
    
    # data from dataframe
    df_index = input_dict['df_index']
    datrow = input_dict['datrow']

    # load coupling prediction models
    klrmodel, kstimmodel, aucintmodel, aucconmodel = coupling_prediction_models()

    # check patient-specific model exists for given patient
    if (datrow['npecode'] not in klrmodel.keys()) or (datrow['npecode'] not in kstimmodel.keys()):
        print(f"Patient missing: Index {df_index} of {datrow['npecode']}.")
        return None

    if np.isnan(datrow['K_left_cday']):
        print(f"K attractor is NaN: Index {df_index} of {datrow['npecode']}.")
        return None

    # patient-specific models
    pat_klrmodel = klrmodel[datrow['npecode']]
    pat_kstimmodel = kstimmodel[datrow['npecode']]
    
    # initialize alternate stimulation instance
    altsimu = kuramotoattractor.AlternateStimulation(datrow=datrow, klrmodel=pat_klrmodel, kstimmodel=pat_kstimmodel, 
                                                     aucintdict=aucintmodel, auccondict=aucconmodel)
    
    # obtain peak sync for different peak hours of stimulation using Kuramoto model
    sampasdf = []   # sample's alternate stimulation dataframe
    for stimhour in np.arange(0, 24):  # 24 hours
        kmindict, alt_peaksync, osppdf, ssterm = altsimu.kuramoto_sol_altstim(stimhour=stimhour)
        store_dict = {'sampleindex': df_index, 'stimhour':stimhour, 'aspeaksync':alt_peaksync, 'ssterm':ssterm} | kmindict
        sampasdf.append( pd.DataFrame(store_dict, index=[0]) )
        
    # convert to dataframe and store
    sampasdf = pd.concat(sampasdf, axis=0, ignore_index=True)

    # save the dataframe
    save_path = f'data/processed/kuramotoattractor/altstimpersample/sindex_%03d.csv' % (df_index)
    sampasdf.to_csv(save_path, index=False)
    
    print(f'Finished index: {df_index}') 

    return None



if __name__ == "__main__":

    print('In main.')
    
    regdf = cycle_info_dataframe()
    
    ###### 
    args_list = [{'df_index':df_index, 'datrow':datrow} for df_index, datrow in regdf.loc[101:104].iterrows()]

    pool = multiprocessing.Pool(processes=2)
    recresult_list = pool.map(alt_stim_for_one_index, args_list)
    
    pool.close()
    print('Pool computation started')
    pool.join()
    print('Pool computation ended')

    
