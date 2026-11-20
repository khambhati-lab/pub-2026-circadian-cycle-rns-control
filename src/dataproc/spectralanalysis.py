import pandas as pd
import numpy as np
import datetime
import scipy
import pickle

from astropy.timeseries import LombScargle
from src.dataproc import utils

def lscircadian_different_durs(patient_id, featdf, feature_list, rangetype='total'):

    # time points to compute LS for
    max_half_year = np.ceil(2 * featdf['tfromstart_days'].max() / 365)
    day_limits = np.round(np.arange(1, max_half_year + 1) * 365 / 2)

    dfcols = ['%s_%d'%(col, day)
              for day in day_limits for col in ['amp_unnorm', 'amp_norm', 'peakhour', 'fitpower',
                                                'pvalue', 'nsamp','hrcoverage']]
    dfcols.insert(0,'feature')
    dfcols.insert(0,'npecode')

    # LS parameters dict
    ccsdf = {key:[] for key in dfcols}

    for featname in feature_list:

        ccsdf['npecode'].append(patient_id)
        ccsdf['feature'].append(featname)

        # LS of feature
        for ii in range(len(day_limits)):

            time_limit = day_limits[ii]

            if rangetype == 'total':

                time_vals = featdf.loc[featdf['tfromstart_days'] <= time_limit]['Round_local_hour'].values
                feat_vals = featdf.loc[featdf['tfromstart_days'] <= time_limit][featname].values

            elif rangetype == 'interval':
                if ii == 0:
                    prev_time_limit = 0
                else:
                    prev_time_limit = day_limits[ii-1]

                time_vals = featdf.loc[(featdf['tfromstart_days'] > prev_time_limit) & (featdf['tfromstart_days'] <= time_limit)]['Round_local_hour'].values
                feat_vals = featdf.loc[(featdf['tfromstart_days'] > prev_time_limit) & (featdf['tfromstart_days'] <= time_limit)][featname].values

            else:
                print(f'{rangetype} not implemented.')
                raise NotImplemented

            
            if len(time_vals) <= 3:
                # cannot compute LS with at most 3 samples with an error/warning. Skip this time limit.
                # store in dict
                ccsdf['amp_unnorm_%d'%(time_limit)].append(np.NaN)
                ccsdf['amp_norm_%d'%(time_limit)].append(np.NaN)
                ccsdf['peakhour_%d'%(time_limit)].append(np.NaN)
                ccsdf['fitpower_%d'%(time_limit)].append(np.NaN)
                ccsdf['pvalue_%d'%(time_limit)].append(np.NaN)
                ccsdf['nsamp_%d'%(time_limit)].append(len(time_vals))
                ccsdf['hrcoverage_%d'%(time_limit)].append(len(np.unique(time_vals)))
                continue

            lsp_object = LombScargle(t=time_vals, y=feat_vals, fit_mean=True)

            lsp_offset = utils.lsmodel_phase_offset(lsp_object, frequency=1/24) # phase offset of the LS model (in radians)
            peak_angle = utils.lsmodel_phase_offset_peak(lsp_offset) # angle at which LS model reaches highest value (sin (angle) = 1) (in radians)

            ls_amp_unnorm = np.linalg.norm(lsp_object.model_parameters(frequency=1/24)[1:])
            # divide LS amplitude by standard deviation of the feature to obtain LS amplitude of normalized feature
            ls_amp_norm = ls_amp_unnorm / np.std(feat_vals)

            # fit diagnostics
            fit_power = lsp_object.power(frequency=1/24)  # normalized power in circadian cycle

            try: # compute p-value if possible
                fit_pvalue = lsp_object.false_alarm_probability(fit_power)  # false alarm probability of given power

            except ValueError: # otherwise, set p-value to 1 and print the power
                fit_pvalue = 1.0
                print(f'>>>>{patient_id}: {featname} for {time_limit} has power {fit_power}.')

            # store in dict
            ccsdf['amp_unnorm_%d'%(time_limit)].append(ls_amp_unnorm)
            ccsdf['amp_norm_%d'%(time_limit)].append(ls_amp_norm)
            ccsdf['peakhour_%d'%(time_limit)].append(24 * peak_angle / (2*np.pi) )
            ccsdf['fitpower_%d'%(time_limit)].append(fit_power)
            ccsdf['pvalue_%d'%(time_limit)].append(fit_pvalue)
            ccsdf['nsamp_%d'%(time_limit)].append(len(time_vals))
            ccsdf['hrcoverage_%d'%(time_limit)].append(len(np.unique(time_vals)))

    # convert to dataframe
    ccsdf = pd.DataFrame(ccsdf)

    return ccsdf