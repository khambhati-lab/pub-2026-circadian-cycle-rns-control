import numpy as np
import pandas as pd
import datetime
from scipy import signal
import os

import src.dataproc.neuropaceIO as neuropaceIO
from src.dataproc import stimartifact
from src.dataproc import edgecases
import pdb

from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__) # log warnings

def spectral_synchrony_wrapper(signalmat, featname, params, band_filters, warninfo=None, breakpoint_flag=False):
    '''
    Compute spectral synchrony between channels for given signal. The algo backbone is code provided by Ankit Khambhati.
    Phase locking value is a measure of spectral synchrony between channels. Hilbert transform of filtered signal is used
    to compute spectral synchrony.

    Parameters:
        signalmat: numpy array of signal amplitude over time for different channels [size: time x channels]
        featname: name of the feature to be computed. 'plv' for phase locking value.
        featparams: dictionary with parameters for feature extraction.
            Must contain keys - channelpairslist, normalize, bands, band_filters, remove_stim_artifacts, cross_freq_flag.
            If stim artifacts are to be removed, then the dictionary should also contain parameters needs for stim artifact removal.
            See function remove_stim_artifacts_from_signal() for details.
        breakpoint_flag: Boolean to indicate if breakpoint should be included or not. Default is False.

    Returns:
        Dataframe of spectral synchrony features for different channel pairs and bands.
            Features values are in columns 'Ch_{channel number 1}-{channel number 2}_{band}' format. E.g., 'Ch_1-Ch_2_theta'.

    Note: Normalization not implemented yet (July 1st 2024).
    Date, Author: --, Krishnakant V. Saboo
    '''

    sigpairslist = params.preproc.channelpairslist # pairs of channels for computing synchrony

    # normalize the signal
    if params.preproc.normalize:
        # normalize the signal
        pass
    else: # else don't normalize
        signalmat_norm = signalmat.copy()


    # filtered signal for computing features: time x band x channel
    signalmat_norm_filt = np.zeros((signalmat_norm.shape[0], len(params.preproc.bands),
        signalmat_norm.shape[1]), dtype=complex) # specifying dtype is important otherwise imaginary part of Hilbert transform is discarded.

    if signalmat_norm.shape[0] <= params.preproc.ntmin:
        logger.debug(f'Filter and Hilbert not possible for {warninfo}.\n')
        signalmat_norm_filt[:,:,:] = np.NaN

    else:
        ##### filter and hilbert transform
        for bn in range(len(params.preproc.bands)): # iterate over bands

            band = params.preproc.bands[bn] # band name
            if band == 'broadband': # if broadband data, then no need for filtering
                signalmat_norm_filt[:,bn,:] = signal.hilbert(signalmat_norm.copy(), axis=0) # apply transform along time axis

            else: # filter the data
                sos = band_filters[band] # filter of the corresponding band and then find analytic signal
                for jj in range(signalmat_norm.shape[1]): # iterate over channels
                    signalmat_norm_filt[:,bn,jj] = signal.hilbert(signal.sosfiltfilt(sos, signalmat_norm[:, jj]))

    ### remove time points corresponding to stimulation artifacts + side
    if params.stim.remove_stim_artifacts:
        signalmat_norm_filt, stimart_times = stimartifact.remove_stim_artifacts_from_signal(signalmat=signalmat, signalmat_filtered=signalmat_norm_filt,
                                                                stimartparams=params.stim)

    ##### compute feature #####
    syncfeat = spectral_synchrony(signal=signalmat_norm_filt, metric=featname,
                                  cross_freq=params.preproc.cross_freq_flag)

    outdict = {} # dictionary for returning feature
    for bn in range(len(params.preproc.bands)):
        band = params.preproc.bands[bn]
        for epn in range(len(sigpairslist)):
            epair = sigpairslist[epn]
            keyname = f'Ch_{epair[0]+1}-Ch_{epair[1]+1}_{band}'
            outdict[keyname] = syncfeat[bn, epair[0], epair[1]]

    # include stimulation information
    outdict['stim_count'] = len(stimart_times)
    if signalmat.shape[0] > 0:
        outdict['frac_timepoints_retained'] = signalmat_norm_filt.shape[0] / signalmat.shape[0]
    else:
        outdict['frac_timepoints_retained'] = 0
    
    # break point for plotting data
    if breakpoint_flag:
        pdb.set_trace()

    # return as series
    return pd.DataFrame(outdict, index=[0])



def spectral_synchrony(signal, metric=None, cross_freq=False):
    """
    Compute inter-electrode synchrony of the iEEG signal based on its spectral
    characteristics. Author: Ankit Khambhati.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Multi-electrode iEEG Signal that has been transformed to the
        complex-domain (via. Hilbert or Wavelet).

    metric: ['plv', 'iplv', 'ciplv', 'coh', 'icoh', 'pli', 'wpli', 'debpli]
        Must supply a metric for calculating spectral synchrony.
        plv - phase locking value (Bruña, R. et. al., 2018).
        iplv - imaginary phase locking value (Bruña, R. et. al., 2018).
        ciplv - corrected imaginary phase locking value (Bruña, R. et. al., 2018).
        coh - coherence (Bruña, R. et. al., 2018).
        icoh - imaginary coherence (Bruña, R. et. al., 2018).
        wpli - weighted phase lag index (Vinck et al., 2011)
        debwpli - debiased weighted phase lag index (Vinck et al., 2011)

    cross_freq: bool
        Whether to retain the cross-frequency interactions, also known as
        the cross-frequency coupling.
        Default is False.

    Returns
    -------
    if cross_freq is False:
        X: numpy.ndarray (Complex), shape: [n_freqs x n_chan x n_chan]
            Synchrony between channels per frequency.
    else:
        X: numpy.ndarray (Complex), shape: [n_freqs x n_freqs x n_chan x n_chan]
            Synchony between channels and between frequencies.
    """

    # Get axes sizes
    n_ts, n_wv, n_ch = signal.shape

    X_cn = signal.copy()

    # Pre-compute the  magnitude
    X_a = np.abs(X_cn)

    # Compute numerator
    if metric in ['plv', 'iplv', 'ciplv']:
        E_XY_tdot = np.tensordot(X_cn/X_a, np.conj(X_cn/X_a), axes=((0), (0))) / n_ts
    elif metric in ['coh', 'icoh', 'wpli']:
        E_XY_tdot = np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0))) / n_ts
    elif metric in ['pli']:
        E_XY_tdot = np.zeros((n_wv, n_ch, n_wv, n_ch))
        for x_cn in X_cn:
            E_XY_tdot += np.sign(np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag)
        E_XY_tdot /= n_ts
    elif metric in ['debwpli']:
        pass
    else:
        raise NotImplemented('Provided metric, {:s}, is unsupported.'.format(metric))

    # Compute denominator
    if metric in ['plv', 'iplv', 'pli']:
        E_XX_tdot = np.ones_like(E_XY_tdot)
    elif metric in ['ciplv']:
        E_XX_tdot = np.sqrt(1 - np.abs((E_XY_tdot.real) / n_ts)**2)
    elif metric in ['coh', 'icoh']:
        E_XX_tdot = np.sqrt(np.tensordot(
            np.expand_dims((X_a**2).mean(axis=0), axis=0),
            np.expand_dims((X_a**2).mean(axis=0), axis=0),
            axes=((0), (0))))
    elif metric in ['wpli']:
        E_XX_tdot = np.zeros((n_wv, n_ch, n_wv, n_ch))
        for x_cn in X_cn:
            E_XX_tdot += np.abs(np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag)
        E_XX_tdot /= n_ts
    elif metric in ['debwpli']:
        pass

    # Compute final metric
    if metric in ['iplv', 'ciplv', 'icoh', 'wpli']:
        E_XY_tdot = np.abs(E_XY_tdot.imag / E_XX_tdot)
    elif metric in ['plv', 'coh']:
        E_XY_tdot = np.abs((E_XY_tdot / E_XX_tdot))
    elif metric in ['pli']:
        E_XY_tdot = np.abs(E_XY_tdot / E_XX_tdot)
    elif metric in ['debwpli']:
        sum_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))
        sum_abs_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))
        sum_sq_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))

        for x_cn in X_cn:
            tdot = np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag
            sum_im_csd += tdot
            sum_abs_im_csd += np.abs(tdot)
            sum_sq_im_csd += np.abs(tdot**2)

        numer = sum_im_csd**2 - sum_sq_im_csd
        denom = sum_abs_im_csd**2 - sum_sq_im_csd
        z_denom = denom == 0
        denom[z_denom] = 1
        E_XY_tdot = numer / denom
        E_XY_tdot[z_denom] = 0
    else:
        return None
    E_XY_tdot = np.nan_to_num(E_XY_tdot)

    # Rearrange the tensor and remove cross-frequency components if desired
    E_XY_tdot =  np.transpose(E_XY_tdot, (0, 2, 1, 3))

    # Remove off-diagonal entries if not considering cross-frequency coupling
    if not cross_freq:
        E_XY_tdot = E_XY_tdot[np.arange(n_wv), np.arange(n_wv), :, :]

    return E_XY_tdot



def power_in_band(signalmat, featname, params, band_filters, warninfo=None, breakpoint_flag=False):

    # normalize the signal
    if params.preproc.normalize:
        # normalize the signal
        pass
    else: # else don't normalize
        signalmat_norm = signalmat.copy()


    # filtered signal for computing features: time x band x channel
    signalmat_norm_filt = np.zeros((signalmat_norm.shape[0], len(params.preproc.bands), signalmat_norm.shape[1]))

    if signalmat_norm.shape[0] <= params.preproc.ntmin:
        logger.debug(f'Filter and Hilbert not possible for {warninfo}.\n')
        signalmat_norm_filt[:,:,:] = np.NaN

    else:
        ##### filter the signal
        for bn in range(len(params.preproc.bands)): # iterate over bands

            band = params.preproc.bands[bn] # band name
            if band == 'broadband': # if broadband data, then no need for filtering
                signalmat_norm_filt[:,bn,:] = signalmat_norm.copy()

            else: # filter the data
                sos = band_filters[band] # filter of the corresponding band and then find filtered signal
                for jj in range(signalmat_norm.shape[1]): # iterate over channels
                    signalmat_norm_filt[:,bn,jj] = signal.sosfiltfilt(sos, signalmat_norm[:, jj])

    ### remove time points corresponding to stimulation artifacts + side
    if params.stim.remove_stim_artifacts:
        signalmat_norm_filt, stimart_times = stimartifact.remove_stim_artifacts_from_signal(signalmat=signalmat, signalmat_filtered=signalmat_norm_filt,
                                                                stimartparams=params.stim)

    ##### compute feature #####
    pibfeat = (signalmat_norm_filt ** 2).mean(axis=0)

    outdict = {} # dictionary for returning feature
    for bn in range(len(params.preproc.bands)):
        band = params.preproc.bands[bn]
        for chn in range(pibfeat.shape[1]):
            keyname = f'Ch_{chn+1}_{band}'
            outdict[keyname] = pibfeat[bn, chn]

    # include stimulation information
    outdict['stim_count'] = len(stimart_times)
    if signalmat.shape[0] > 0:
        outdict['frac_timepoints_retained'] = signalmat_norm_filt.shape[0] / signalmat.shape[0]
    else:
        outdict['frac_timepoints_retained'] = 0
    
    # break point for plotting / checking data
    if breakpoint_flag:
        pdb.set_trace()

    # return as series
    return pd.DataFrame(outdict, index=[0])


def extract_ECoG_feature(folder_path, df_catalog_pt, params):
    '''
    Extract features from RNS ECoG.

    Parameters:
        folder_path: path to folder containing RNS ECoG files.
        df_catalog_pt: catalog file containing names of RNS ECoG files of a patient along with timestamp info.
        params: dictionary containing parameters for extracting RNS ECoG.
            Must contain keys - NUMCHAN, FSAMPLING, sigduration, sigtype, featname, featparams (dictionary of feature extraction parameters).

    Returns:
        ptft_catalog_df: dataframe with feature values as columns along with information from catalog file.

    Note: Has functionality to insert breakpoints.
    Date, Author: --, Krishnakant V. Saboo
    '''

    outdf_list = [] # store results

    ## compute filters
    band_filters = {}
    for band in params.preproc.bands:
        if band == 'broadband':
            continue # no filter needed if broadband=all frequencies are needed
        # filter parameters for each band; second order butterworth filter
        # (effective order=4 because of filtfilt during filtering)
        band_filters[band] = signal.butter(N=params.preproc.filter_order, Wn=params.preproc.band_cutoffs[band],
                                           btype=params.preproc.filter_pass, fs=params.data.FSAMPLING,
                                           output=params.preproc.filter_type)


    for epinum in range(df_catalog_pt.shape[0]): # each episode separately
        # get data from an episode
        if epinum % 1000 == 0:
            print(f'Extracting features from episode: {epinum}')

        data_dict = neuropaceIO.get_ieeg_data_dict(df_catalog_pt.iloc[[epinum]], folder_path)
        
        # extract relevant segment of the data
        if params.data.sigtype == 'pretrig': # pre-trigger data
            
            # obtain the pre-trigger length
            pretrigsiglen = int(df_catalog_pt['ECoG pre-trigger length'].iloc[epinum] * params.data.FSAMPLING)
            
            if params.data.sigduration == 'full': # duration before trigger to include
                seglen = pretrigsiglen
            else:
                seglen = min(pretrigsiglen, params.data.sigduration * params.data.FSAMPLING) # number of points to use for feature
            
            # read the pre-trigger data into a matrix
            ptsmat = np.zeros((seglen, params.data.NUMCHAN))
            for ii in range(params.data.NUMCHAN):
                ptsmat[:,ii] = data_dict['signal'][pretrigsiglen - seglen : pretrigsiglen, ii]

        elif params.data.sigtype == 'full':
            ptsmat = data_dict['signal'].copy() # take the entire signal

        else:
            print('sigtype did not match known types.')

        # compute feature
        if params.data.featname == 'plv': # phase locking value between channel pairs
            oeoutdf = spectral_synchrony_wrapper(ptsmat, params.data.featname, params, band_filters, warninfo=df_catalog_pt['Filename'].iloc[epinum]) # synchrony between features for one episode
            oeoutdf['Timestamp_utc'] = df_catalog_pt['Timestamp_utc'].iloc[epinum] # to match results with catalog file
            outdf_list.append(oeoutdf)

        elif params.data.featname == 'pib': # power in band feature
            # call function to compute power in band feature and store results
            oeoutdf = power_in_band(ptsmat, params.data.featname, params, band_filters, warninfo=df_catalog_pt['Filename'].iloc[epinum], breakpoint_flag=False) # power in band for one episode
            oeoutdf['Timestamp_utc'] = df_catalog_pt['Timestamp_utc'].iloc[epinum] # to match results with catalog file
            outdf_list.append(oeoutdf)

        else:
            print('feature type is not available.')

    # outside for loop; concatenate dfs from all episodes
    featdf = pd.concat(outdf_list, axis=0, ignore_index=True)

    # include feature info into original dataframe
    ptft_catalog_df = df_catalog_pt.merge(featdf, left_on='Timestamp_utc', right_on='Timestamp_utc')
    
    return ptft_catalog_df



def extract_ECoG_feature_wrapper(patient_code, pathsdict, paramsdict):
    '''
    Loads the catalog file containing ECoG paths for given patient. Passes the catalog df along to
    appropriate feature extraction function.

    Parameters:
        patient_code: NP code of patient. Should be present in the catalog file.
        pathsdict: Dictionary of paths. Must contain these keys - df_catalog_path, ecog_folder_path, save_catalog_path
        paramsdict: Dictionary of parameters. Keys depend on the features to be extracted.

    Returns:
        None. Saves the feature file based on the path provided in pathsdict.

    Date, Author: --, Krishnakant V. Saboo
    '''

    # load dataframes of catalog files for given NPE code
    if os.path.exists(pathsdict.df_catalog_path):
        df_catalog = pd.read_csv(pathsdict.df_catalog_path) # catalog file
    else:
        logger.debug(f'Catalog file not found: {pathsdict.df_catalog_path}.')
        print(f'Catalog file not found: {pathsdict.df_catalog_path}.')
        return

    # transform date into timestamp
    df_catalog['Timestamp_utc'] = df_catalog['Raw UTC timestamp'].apply(lambda x: datetime.datetime.fromisoformat(x).timestamp())

    # get subset of catalog related to given trigger type (eg Scheduled, Long_Episode)
    df_catalog_pt = df_catalog.loc[(df_catalog['ECoG trigger'] == paramsdict.data.trigger)].copy()

    # handle any edge cases in catalog file
    df_catalog_pt = edgecases.catalogdf_edgecases(patient_code, df_catalog_pt)

    # compute feature
    ptft_catalog_df = extract_ECoG_feature(folder_path=pathsdict.ecog_folder_path, df_catalog_pt=df_catalog_pt,
                                           params=paramsdict)

    # save the result
    ptft_catalog_df.to_csv(pathsdict.save_feature_path, index=False)



def main_featext(input_dict):
    '''
    Main function to extract spectral synchrony in band feature for a given patient and save the resulting dataframe.
    Includes functionality to remove stimulation artifacts from ECoG before computing feature.

    Parameters:
        input_dict: Dictionary containing patient ID 'npcode' and machine to implement feature extraction ('local' or 'cluster').

    Returns:
        None. Saves the resulting dataframe based on the paths in the code.

    Note: Updated to include functionality to remove stimulation artifacts from ECoG before computing feature (last week of June 2024).
    Date, Author: --, Krishnakant V. Saboo
    '''

    # parse input from input dictionary
    machine, patient_code = input_dict['machine'], input_dict['npcode']
    print(f'In cluster implementation for {patient_code}')

    ##### parameters
    params = OmegaConf.load('src/conf/features/plv_perhz.yaml')

    ##### paths
    paths = OmegaConf.load(f'src/conf/paths/{machine}.yaml')
    paths.df_catalog_path = paths.df_catalog_path % (patient_code, patient_code)
    paths.ecog_folder_path = paths.ecog_folder_path % (patient_code, patient_code)
    paths.save_feature_path = paths.save_feature_path % (params.data.dirprefix, patient_code, params.data.trigger)

    ##### compute feature
    extract_ECoG_feature_wrapper(patient_code=patient_code, pathsdict=paths,  paramsdict=params)





if __name__ == '__main__':
    logging.basicConfig(filename=f"logs/log_{datetime.datetime.now()}.txt", level=logging.DEBUG)
    logging.captureWarnings(True)
    logging.info(f'Running from src/dataproc/rnsfeatureext')

    main_featext({'machine':'cluster', 'npcode':'NPE0140'})