import pandas as pd
import numpy as np
from astropy.stats.circstats import circmean, circvar
from scipy.stats import pearsonr
import statsmodels.regression as sm



def lastavail_day(col_list, day_limit):
    '''Last available day for which LS has been available in the given columns.'''
    
    return max([int(item.split('_')[-1]) for item in col_list 
                if ('nsamp' in item) and (int(item.split('_')[-1]) <= day_limit)])


def days_available(col_list):
    return sorted([int(item.split('_')[-1]) for item in col_list if ('nsamp' in item)])


def aggregate_cycle_params(lsdf_dict, patients, bands, edges, featuredict, day_limit, 
                           colprefix, aggtype='sum'):
    
    aggdf = {key:[] for key in ['npecode', 'band']+[f'{colprefix}_{ed}' for ed in edges]}
    
    for patient_id in patients:
        
        lsdf = lsdf_dict[patient_id].copy()
        
        for bandnm in bands:

            aggdf['npecode'].append(patient_id)
            aggdf['band'].append(bandnm)

            for edtype in edges:
                lastday = lastavail_day(lsdf.columns, day_limit)
                tdf = lsdf.loc[lsdf['feature'].isin(featuredict[edtype][bandnm])].copy()
                cycle_params = tdf[f'{colprefix}_{lastday}'].to_numpy()
                
                if aggtype == 'sum':
                    # compute sum of LS amplitudes across features
                    # aggdf[f'{colprefix}_{edtype}'].append(tdf[f'{colprefix}_{lastday}'].sum())
                    aggdf[f'{colprefix}_{edtype}'].append( np.sum(cycle_params) )
                    
                elif aggtype == 'circvar':
                    # circspread = circvar( tdf[f'{colprefix}_{lastday}'].to_numpy() * 2 * np.pi/24 )
                    circspread = 1 - circvar( cycle_params * 2 * np.pi/24 ) # convert to synchronization
                    aggdf[f'{colprefix}_{edtype}'].append(circspread)
    
    # convert to dataframe
    aggdf = pd.DataFrame(aggdf)
    
    return aggdf


def longi_aggregate_cycle_params(lsdf_dict, patients, bands, edges, featuredict, colprefix, aggtype='sum'):
    
    aggdf = {key:[] for key in ['npecode', 'days_from_implant', 'band']+[f'{colprefix}_{ed}' for ed in edges]}
    
    for patient_id in patients:
        
        lsdf = lsdf_dict[patient_id].copy()
        pat_days_list = days_available(lsdf.columns)
        
        for daynum in pat_days_list:
        
            for bandnm in bands:

                aggdf['npecode'].append(patient_id)
                aggdf['days_from_implant'].append(daynum)
                aggdf['band'].append(bandnm)

                for edtype in edges:

                    tdf = lsdf.loc[lsdf['feature'].isin(featuredict[edtype][bandnm])].copy()
                    cycle_params = tdf[f'{colprefix}_{daynum}'].to_numpy()

                    if aggtype == 'sum':
                        # compute sum of LS amplitudes across features
                        # aggdf[f'{colprefix}_{edtype}'].append(tdf[f'{colprefix}_{daynum}'].sum())
                        aggdf[f'{colprefix}_{edtype}'].append( np.sum(cycle_params) )
                    
                    elif aggtype == 'circvar':
                        # circspread = circvar( tdf[f'{colprefix}_{lastday}'].to_numpy() * 2 * np.pi/24 )
                        circspread = 1 - circvar( cycle_params * 2 * np.pi/24 )
                        aggdf[f'{colprefix}_{edtype}'].append(circspread)
    
    # convert to dataframe
    aggdf = pd.DataFrame(aggdf)
    
    return aggdf


def closest_outcome_day(row, odf, timelocktype):
    odays = odf.loc[odf['npe_code'] == row['npecode']]['outcome_days'].to_numpy()
    
    if len(odays) == 0:
        return np.NaN
    
    if timelocktype == 'causal':
        odays_diff = np.array(odays - row['feat_days'], dtype=float)
        odays_diff[odays_diff < 0] = np.inf   # anything in the future is set to inf
        
        if np.isfinite(odays_diff).any():  # check if there is any outcome after the current feature date
            return int(odays[np.argmin( odays_diff )])  # outcome day closest to feature collection that is after feature collection
        else:
            return np.NaN  # else, return nan because no outcome is after current feature collection date
            
    else:
        return odays[np.argmin( np.abs(odays - row['feat_days']) )]  # closest causal or noncausal day


def closest_outcome_and_feature_df(outdf, featdf, patient_list, timelock_type='noncausal', verbose=False):
    # create a copy of the outcome 
    regdf = outdf.loc[outdf['npe_code'].isin(patient_list)][['npe_code', 'baseline_seizure_rate', 'days_from_implant', 'sz_red']].copy()
    regdf.rename(columns={'days_from_implant':'outcome_days'}, inplace=True)
    
    # feature dataframe copy
    regdf_feat = featdf.copy()
    regdf_feat.rename(columns={'days_from_implant':'feat_days'}, inplace=True)

    # find the outcome closest to the feature computation day
    regdf_feat['closest_outcome_days'] = regdf_feat.apply(lambda x: closest_outcome_day(x, regdf, timelock_type), axis=1)
    
    # drop nans - where outcome and feature could not be matched
    prenansize = regdf_feat.shape[0]
    regdf_feat.dropna(subset='closest_outcome_days', inplace=True)
    postnansize = regdf_feat.shape[0]
    
    if verbose:
        print(f'Dropped nans, where feature and outcome couldnt be matched: # {prenansize - postnansize}')
    
    num_dup_pre_filter = regdf_feat.loc[regdf_feat['band']=='alpha'].duplicated(subset=['npecode', 'closest_outcome_days']).sum()
    
    if verbose:
        print(f'Number of duplicate entries in alpha outcome before filtering: {num_dup_pre_filter}')
    
    # multiple feature computations days can be linked to the same outcome but only one of them in closest.
    # find the closest one and only retain that
    regdf_feat['outcome_feat_gap'] = np.abs(regdf_feat['closest_outcome_days'] - regdf_feat['feat_days'])

    idxdf = regdf_feat.groupby(by=['npecode', 'closest_outcome_days'])['outcome_feat_gap'].min().reset_index()

    regdf_feat_ss = []
    for idx, row in idxdf.iterrows():
        tdf = regdf_feat.loc[(regdf_feat['npecode'] == row['npecode']) & 
                             (regdf_feat['closest_outcome_days']== row['closest_outcome_days']) & 
                             (regdf_feat['outcome_feat_gap'] == row['outcome_feat_gap'])].copy()
        regdf_feat_ss.append(tdf)

    # convert to dataframe
    regdf_feat = pd.concat(regdf_feat_ss, ignore_index=True, axis=0)
    
    num_dup_post_filter = regdf_feat.loc[regdf_feat['band']=='alpha'].duplicated(subset=['npecode', 'closest_outcome_days']).sum()
    if verbose:
        print(f'Number of duplicate entries in alpha outcome after filtering: {num_dup_post_filter}')
    
    # merge features with outcome information
    regdf = regdf_feat.merge(regdf, left_on=['npecode', 'closest_outcome_days'], right_on=['npe_code', 'outcome_days'])
    
    return regdf


def round_half_year_days(indf, colname='tfromstart_days', max_time_yr=20):
    
    day_limits = np.round(np.arange(0, 2*max_time_yr + 1) * 365 / 2)  # 6 month intervals

    def _closest_upper_bound(myarr, myval):
        mydiff = myarr - myval
        mydiff[mydiff < 0] = np.inf
        return int(myarr[np.argmin(mydiff)])

    indf['dur_halfyr'] = indf[colname].apply(lambda x: _closest_upper_bound(day_limits, x) )
    
    return indf


def mean_stimcounts(dfdict, patient_list, groupby, normtype='zeroone'):
    """ Only valid groupby values for now = ['Local_hour'] and ['dur_halfyr', 'Local_hour'] """
    
    msmnormdf = {} # mean stim max norm df
    
    for patient_id in patient_list:

        df = dfdict[patient_id].copy()

        # normalized avg stim counts
        mean_stim_count = df.groupby(by=groupby)['Episode starts with RX'].mean().reset_index()
        mean_stim_count.rename(columns={'Episode starts with RX':'mean_sc'}, inplace=True)
        
        if groupby == ['dur_halfyr', 'Local_hour']:
            mean_stim_count = mean_stim_count.pivot(index=['Local_hour'], columns=['dur_halfyr'], values=['mean_sc'])
            
            # rename columns of the pivot table
            mean_stim_count.columns = mean_stim_count.columns.to_flat_index()
            mean_stim_count.rename(columns=dict(zip(mean_stim_count.columns, 
                                                    ['%s-%d'%(item) for item in mean_stim_count.columns])), 
                                   inplace=True)
            
            # drop any nan columns
            mean_stim_count.dropna(axis=1, inplace=True)
            
        elif groupby == ['Local_hour']:
            mean_stim_count.set_index('Local_hour', inplace=True)
            
        else:
            raise NotImplemented
            
        # normalize the data
        if normtype == 'maxnorm':
            mean_stim_count_norm = mean_stim_count / mean_stim_count.max()
            
            # if mean stim is zero at all times, then normalization will gives NaNs. Address the nans by replacing
            # zero stim with 0 or 1. If stim if 0, number of unique entires in the column = 1.
            for col in mean_stim_count.columns:
                if (mean_stim_count[col].nunique() == 1) and (mean_stim_count[col].unique() == 0):
                    mean_stim_count_norm[col] = 0.
                    
                elif (mean_stim_count[col].nunique() == 1) and (mean_stim_count[col].unique() != 0):
                    mean_stim_count_norm[col] = 1.
                
                # else; do nothing
                
        
        elif normtype == 'zeroone':
            mean_stim_count_norm = (mean_stim_count - mean_stim_count.min()) / (mean_stim_count.max() - mean_stim_count.min())
        
            # if mean stim is zero at all times, then normalization will gives NaNs. Address the nans by replacing
            # zero stim with 0. If stim if 0, number of unique entires in the column = 1.
            for col in mean_stim_count.columns:
                if mean_stim_count[col].nunique() == 1:
                    mean_stim_count_norm[col] = 0.

        elif normtype == 'density':
            mean_stim_count_norm = mean_stim_count / mean_stim_count.sum()

            # if mean stim is zero at all times, then normalization will gives NaNs. Address the nans by replacing
            # zero stim with 0. If stim if 0, number of unique entires in the column = 1.
            for col in mean_stim_count.columns:
                if mean_stim_count[col].nunique() == 1:
                    mean_stim_count_norm[col] = 0.

        elif normtype == 'zeroone-density':
            mean_stim_count_norm = (mean_stim_count - mean_stim_count.min()) / (mean_stim_count.max() - mean_stim_count.min())
            mean_stim_count_norm = mean_stim_count_norm / mean_stim_count_norm.sum()
            
            # if mean stim is zero at all times, then normalization will gives NaNs. Address the nans by replacing
            # zero stim with 0. If stim if 0, number of unique entires in the column = 1.
            for col in mean_stim_count.columns:
                if mean_stim_count[col].nunique() == 1:
                    mean_stim_count_norm[col] = 0.

        elif normtype == 'zscore':
            mean_stim_count_norm = (mean_stim_count - mean_stim_count.mean()) / mean_stim_count.std()
            
            # if mean stim is zero at all times, then normalization will gives NaNs. Address the nans by replacing
            # zero stim with 0. If stim if 0, number of unique entires in the column = 1.
            for col in mean_stim_count.columns:
                if mean_stim_count[col].nunique() == 1:
                    mean_stim_count_norm[col] = 0.

        
        elif normtype == 'nonorm':
            mean_stim_count_norm = mean_stim_count  # no normalization

        else:
            raise NotImplemented
        
        msmnormdf[patient_id] = mean_stim_count_norm.copy()
        
    return msmnormdf


def unroll_df_in_time(indf, timecol, num_rolls=2, period_per_roll=24):
    
    repdf_list = []
    
    for ii in range(num_rolls):
        repdf = indf.copy()
        repdf[f'{timecol}_unroll'] = repdf[timecol] + ii * period_per_roll   # unrolled version of time
        repdf_list.append(repdf)
        
    retrepdf = pd.concat(repdf_list, axis=0, ignore_index=True)
    
    return retrepdf


def interval_aucdf(datadfdict, patient_list, interval_start, interval_end):
    
    aucdf = {key:[] for key in ['npe_code', 'auc_interval', 'auc_converse']}
    
    for patient_id in patient_list:
        stimcount = datadfdict[patient_id].copy()
        auc_all = stimcount.sum().item()
        auc_interval = stimcount[interval_start:interval_end].sum().item()
        auc_converse = auc_all - auc_interval
        
        # store results
        aucdf['npe_code'].append(patient_id)
        aucdf['auc_interval'].append(auc_interval)
        aucdf['auc_converse'].append(auc_converse)

    # convert to dataframes
    aucdf = pd.DataFrame(aucdf)
    return aucdf



def longi_interval_aucdf(datadfdict, patient_list, interval_start, interval_end, verbose=False):

    lgaucdf = []

    for patient_id in patient_list:

        lgmscdf = datadfdict[patient_id].copy()

        # compute AUC over time
        auc_all = lgmscdf.sum()
        auc_interval = lgmscdf.loc[interval_start:interval_end].sum()
        auc_converse = auc_all - auc_interval
        max_hours = lgmscdf.idxmax()   # hour at which max mean stim happens

        # concat dataframe and format the columns
        pat_aucdf = pd.concat((auc_all, auc_interval, auc_converse, max_hours), 
                              axis=1, ignore_index=False)
        pat_aucdf.rename(columns={0:'auc', 1:'auc_interval', 2:'auc_converse', 3:'max_hour'}, 
                         inplace=True)
        pat_aucdf.reset_index(names='dur_day', inplace=True)
        pat_aucdf['dur_day'] = pat_aucdf['dur_day'].apply(lambda x: int(x.split('-')[-1]) )
        pat_aucdf.insert(0, 'npecode', patient_id)
        
        # append
        lgaucdf.append(pat_aucdf)
        
    lgaucdf = pd.concat(lgaucdf, axis=0, ignore_index=True)
    lgaucdf.drop(index=lgaucdf.index[lgaucdf['dur_day']==0], inplace=True)

    # sine and cosine of max hour
    lgaucdf['cos_hour'] = lgaucdf['max_hour'].apply(lambda x: np.cos(np.pi/12 * x) )
    lgaucdf['sin_hour'] = lgaucdf['max_hour'].apply(lambda x: np.sin(np.pi/12 * x) )

    # drop 0 AUC entries
    prezerodropsize = lgaucdf.shape[0]
    lgaucdf.drop(index=lgaucdf.index[lgaucdf['auc']==0], inplace=True)
    postzerodropsize = lgaucdf.shape[0]
    
    if verbose:
        print(f'Dropped entries because stim AUC=0: # {prezerodropsize - postzerodropsize}')

    return lgaucdf




def range_of_hours(myhours):
    
    if len(myhours) <= 1:
        return np.NaN
    
    myhours_arr = myhours.to_numpy()
    myangles = myhours_arr * np.pi / 12
    max_ang_diff = 0

    for ii in range(len(myangles)-1):
        for jj in range(ii,len(myangles)):
            ang1 = myangles[ii]
            ang2 = myangles[jj]
            ang_diff = np.arccos(np.cos(ang1 - ang2))

            if ang_diff > max_ang_diff:
                max_ang_diff = ang_diff

    max_ang_hr = max_ang_diff * 12 / np.pi
    
    return max_ang_hr


def longi_mean_cycle_peakhour(lsdf_dict, patients, bands, edges, featuredict, colprefix='peakhour'):
    
    aggdf = {key:[] for key in ['npecode', 'days_from_implant', 'band']+[f'maxprobhour_{ed}' for ed in edges]+\
    [f'cycle1peak_{ed}' for ed in edges]+[f'cycle2peak_{ed}' for ed in edges]+\
    [f'cycle1amp_{ed}' for ed in edges]+[f'cycle2amp_{ed}' for ed in edges]}
    
    for patient_id in patients:
        
        lsdf = lsdf_dict[patient_id].copy()
        pat_days_list = days_available(lsdf.columns)
        
        for daynum in pat_days_list:
        
            for bandnm in bands:

                aggdf['npecode'].append(patient_id)
                aggdf['days_from_implant'].append(daynum)
                aggdf['band'].append(bandnm)

                for edtype in edges:

                    tdf = lsdf.loc[lsdf['feature'].isin(featuredict[edtype][bandnm])].copy()
                    cycle_params = tdf[f'{colprefix}_{daynum}'].to_numpy()
                    cycle_amp = tdf[f'amp_unnorm_{daynum}'].to_numpy()

                    if len(cycle_params) > 2:
                        raise NotImplemented

                    if np.isnan(cycle_params).any():
                        aggdf[f'maxprobhour_{edtype}'].append(np.NaN)
                        aggdf[f'cycle1peak_{edtype}'].append(np.NaN)
                        aggdf[f'cycle2peak_{edtype}'].append(np.NaN)
                        aggdf[f'cycle1amp_{edtype}'].append(np.NaN)
                        aggdf[f'cycle2amp_{edtype}'].append(np.NaN)
                        continue

                    cc1_prob = (np.cos((np.arange(24) - cycle_params[0]) * np.pi/12) + 1 ) / 2   # (cos(theta) + 1) / 2
                    cc2_prob = (np.cos((np.arange(24) - cycle_params[1]) * np.pi/12) + 1 ) / 2
                    max_prob_hour = np.nanargmax(cc1_prob * cc2_prob)

                    diff_val = min(np.abs(cycle_params[0] - cycle_params[1]), 24 - np.abs(cycle_params[0] - cycle_params[1]))

                    aggdf[f'maxprobhour_{edtype}'].append(max_prob_hour)
                    aggdf[f'cycle1peak_{edtype}'].append(cycle_params[0])
                    aggdf[f'cycle2peak_{edtype}'].append(cycle_params[1])
                    aggdf[f'cycle1amp_{edtype}'].append(cycle_amp[0])
                    aggdf[f'cycle2amp_{edtype}'].append(cycle_amp[1])
    
    # convert to dataframe
    aggdf = pd.DataFrame(aggdf)
    
    return aggdf


def outlier_threshold(datarr):
    p25 = np.percentile(datarr, 25)
    p75 = np.percentile(datarr, 75)
    iqr = p75-p25
    return p25 - 1.5*iqr, p75 + 1.5*iqr


def angle_gap(row, col1, col2):
    return 12/np.pi * np.arccos(np.cos(np.pi/12 * (row[col1] - row[col2])))


def peak_sync_from_hours(hour1, hour2):
    return 1 - circvar(np.pi / 12 * np.array([hour1, hour2]))

def peak_sync_from_angles(ang1, ang2):
    return 1 - circvar(np.array([ang1, ang2]))

def within_12h_clockwise(hr1, hr2):
    wrap_flag = False
    win_start = hr1
    win_end = hr1 + 12
    if win_end > 24:
        win_start2 = 0
        win_end2 = win_end % 24
        win_end = 24
        wrap_flag = True

    if not wrap_flag: # within the same 24 hr period
        if (hr2 >= win_start) & (hr2 <= win_end):
            return 1  # closer in clockwise direction
        else:
            return -1 # closer in anticlockwise direction
    
    else: # between two different 24 hr periods
        if ( (hr2 >= win_start) & (hr2 <= win_end) ) | ( (hr2 >= win_start2) & (hr2 <= win_end2) ):
            return 1
        else:
            return -1



def final_patient_list():
    patient_list = ['NPE0001', 'NPE0005', 'NPE0015', 'NPE0017', 'NPE0019', 'NPE0021', 'NPE0029', 
    'NPE0034', 'NPE0036', 'NPE0038', 'NPE0039', 'NPE0041', 'NPE0043', 'NPE0045', 'NPE0047', 
    'NPE0052', 'NPE0055', 'NPE0056', 'NPE0061', 'NPE0062', 'NPE0063', 'NPE0068', 'NPE0070', 
    'NPE0072', 'NPE0086', 'NPE0087', 'NPE0088', 'NPE0091', 'NPE0095', 'NPE0096', 'NPE0099', 
    'NPE0102', 'NPE0103', 'NPE0104', 'NPE0120', 'NPE0123', 'NPE0125', 'NPE0140'] 

    return patient_list


def create_lgfcccfeat_dataframe(lsdf_dict, patient_list, band_list, edge_type_list, subfeatdict):

    # compute the LS amplitude across different feature types for each patient
    lgunampdf = longi_aggregate_cycle_params(lsdf_dict, patient_list, band_list, edge_type_list, 
        subfeatdict, colprefix='amp_unnorm', aggtype='sum')

    lgpkvardf = longi_aggregate_cycle_params(lsdf_dict, patient_list, band_list, edge_type_list, 
        subfeatdict, colprefix='peakhour', aggtype='circvar')

    # obtain feature only dataframe
    lgfeatdf = lgunampdf.merge(lgpkvardf, on=['npecode', 'band', 'days_from_implant'])

    # drop nan rows
    lgfeatdf.dropna(axis=0, inplace=True)

    return lgfeatdf


def create_cop_dataframe(lgpatcodesdf, lgfeatdf, patient_list, mreg_id_vars, mreg_value_vars, 
    tltype='causal', verbose=False):

    # associate longitudinal outcomes and features
    mregdf = closest_outcome_and_feature_df(lgpatcodesdf, lgfeatdf, patient_list, tltype)

    predropnasize = mregdf.shape[0]
    mregdf.dropna(inplace=True)  
    postdropnasize = mregdf.shape[0]
    if verbose: print(f'Dropped nans in regdf: {predropnasize - postdropnasize}')

    # check that features were collected before (or upto) outcome for causal merge
    if (tltype == 'causal') and (mregdf.loc[mregdf['outcome_days'] < mregdf['feat_days']].shape[0] != 0):
        print('Got non-causal pairs even though we wanted causal pairs only.')
        raise ValueError

    # drop extra columns
    mregdf.drop(columns=['closest_outcome_days', 'outcome_feat_gap', 'npe_code'], inplace=True)

    # pivot columns on band and rename the columns
    mregdf = mregdf.pivot_table(index=mreg_id_vars, columns=['band'], values=mreg_value_vars)
    mregdf.columns = mregdf.columns.to_flat_index()
    mregdf.rename(columns=dict(zip(mregdf.columns, ['-'.join(item) for item in mregdf.columns])), inplace=True)
    mregdf.reset_index(inplace=True) 

    # drop the feature-outcome pairs that are more than 1 yr apart
    mregdf['days_diff'] = np.abs(mregdf['outcome_days'] - mregdf['feat_days'])
    mregdf.drop(index=mregdf.index[mregdf['days_diff'] > 365], inplace=True)

    # drop the feature-outcome pairs that have no stimulation
    mregdf.drop(index=mregdf.index[mregdf['auc'] == 0], inplace=True)

    # # time of max hour as sin and cos
    # mregdf['cos_hour'] = mregdf['max_hour'].apply(lambda x: np.cos( np.pi/12 * x ))
    # mregdf['sin_hour'] = mregdf['max_hour'].apply(lambda x: np.sin( np.pi/12 * x ))

    # seizure rate in log scale
    mregdf['log_baseline_seizure_rate'] = np.log(mregdf['baseline_seizure_rate']) 

    return mregdf


def get_subfeatdict(feature_list, band_list, edge_type_list):

    subfeatdict = {key:{} for key in edge_type_list}

    for edge_type in edge_type_list:
    
        if edge_type == 'local':
            edgetype_feature_list = [item for item in feature_list if ('Ch_1-Ch_2' in item) or ('Ch_3-Ch_4' in item)]

        elif edge_type == 'longrange':
            edgetype_feature_list = [item for item in feature_list if not(('Ch_1-Ch_2' in item) or ('Ch_3-Ch_4' in item))]
    
        else:
            edgetype_feature_list = feature_list
    
        
        for bandnm in band_list + ['all']:
            if (bandnm != 'all'):
                subfeatdict[edge_type][bandnm] = [item for item in edgetype_feature_list if bandnm in item]

            else: # if all, include only theta, alpha, beta, gamma
                subfeatdict[edge_type][bandnm] = [item for item in edgetype_feature_list if ('theta' in item) or ('alpha' in item) or ('beta' in item) or ('gamma' in item)]

    # return the data
    return subfeatdict


def align_curr_previous_fccc(lgfeatdf, pivot_band_into_column=True):

    # copy of dataframe to get the feature values at the previous time point
    lgfeatprevdf = lgfeatdf.copy()

    # dictionary mapping from current to previous timepoint
    day_limits = np.round(np.arange(0, 2*20 + 1) * 365 / 2)
    yearsdict = dict(zip(day_limits[1:], day_limits[:-1]))

    # column used for merging current and previous feature dataframes
    lgfeatdf['days_from_implant_prev'] = lgfeatdf['days_from_implant'].apply(lambda x: int(yearsdict[x]))

    lgfeatprevdf.rename(columns={key:f'{key}_prev' for key in lgfeatprevdf.columns 
        if ('local' in key) or (key=='days_from_implant')}, inplace=True)

    # merged dataframe with current and prev feature values
    lgfeatdf = lgfeatdf.merge(lgfeatprevdf, how='left', 
                              on=['npecode', 'band', 'days_from_implant_prev'])

    if pivot_band_into_column:
        # pivot table and flatten the column names
        lgfeatdf = lgfeatdf.pivot(index=['npecode', 'days_from_implant', 'days_from_implant_prev'], 
            columns=['band'], 
            values=['amp_unnorm_local', 'peakhour_local', 'amp_unnorm_local_prev', 'peakhour_local_prev'])

        lgfeatdf.columns = lgfeatdf.columns.to_flat_index()
        lgfeatdf.rename(columns=dict(zip(lgfeatdf.columns, ['-'.join(item) for item in lgfeatdf.columns])), inplace=True)
        lgfeatdf.reset_index(inplace=True)

    return lgfeatdf


def melt_left_right_into_single_column(lgdf):

    fsdf_1 = lgdf[['npecode', 'dur_day', 
                   'cycle1amp_local', 'cycle1peak_local', 
                   'cycle1peak_local_prev', 'cycle1amp_local_prev', 
                   'max_hour', 'auc', 'auc_converse', 'auc_interval']].copy()
    fsdf_1.rename(columns={'cycle1amp_local':'amp', 'cycle1peak_local':'peaktime', 
                           'cycle1peak_local_prev':'peaktime_prev', 'cycle1amp_local_prev':'amp_prev'}, 
                  inplace=True)
    fsdf_1['cycle'] = 1

    fsdf_2 = lgdf[['npecode', 'dur_day',
                   'cycle2amp_local', 'cycle2peak_local', 
                   'cycle2peak_local_prev', 'cycle2amp_local_prev', 
                   'max_hour', 'auc', 'auc_converse', 'auc_interval']].copy()
    fsdf_2.rename(columns={'cycle2amp_local':'amp', 'cycle2peak_local':'peaktime', 
                           'cycle2peak_local_prev':'peaktime_prev', 'cycle2amp_local_prev':'amp_prev'}, 
                  inplace=True)
    fsdf_2['cycle'] = 2

    fsdf = pd.concat((fsdf_1, fsdf_2), ignore_index=True, axis=0)
    fsdf['fs_gap'] = fsdf.apply(lambda x: angle_gap(x, 'peaktime', 'max_hour'), axis=1)
    fsdf['fs_sgngap'] = fsdf.apply(lambda x: within_12h_clockwise(x['max_hour'], x['peaktime']) * x['fs_gap'], axis=1)

    fsdf['ffprev_gap'] = fsdf.apply(lambda x: angle_gap(x, 'peaktime', 'peaktime_prev'), axis=1)
    fsdf['ffprev_sgngap'] = fsdf.apply(lambda x: within_12h_clockwise(x['peaktime_prev'], x['peaktime']) * x['ffprev_gap'], axis=1)
    fsdf['ffprev_sgn'] = fsdf.apply(lambda x: within_12h_clockwise(x['peaktime_prev'], x['peaktime']), axis=1)

    # sin and cos of time
    fsdf['sin_stim'] = fsdf['max_hour'].apply(lambda x: np.sin( np.pi/12 * x))
    fsdf['cos_stim'] = fsdf['max_hour'].apply(lambda x: np.cos( np.pi/12 * x))

    fsdf['sin_fccc_prev'] = fsdf['peaktime_prev'].apply(lambda x: np.sin( np.pi/12 * x))
    fsdf['cos_fccc_prev'] = fsdf['peaktime_prev'].apply(lambda x: np.cos( np.pi/12 * x))

    return fsdf



