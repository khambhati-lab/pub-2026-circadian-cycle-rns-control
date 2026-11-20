import numpy as np
import pandas as pd
import scipy
from scipy import stats
import datetime
import pickle

import statsmodels.formula.api as smf



def ismonotonicdec(myarr):
    '''
    Check whether an array is monotonically decreasing.
    Parameters:
        myarr: numpy array.
    Returns:
        False if array is monotonically non-decreasing, otherwise True.
    Date, Author: --, Krishnakant V. Saboo
    '''
    if (np.diff(myarr) > 0).any():
        return False
    else:
        return True


def daysfrac(curdt, origindt):
    '''
    Compute number of days since origin. Returns a fraction based on the number of days and hours since origin.

    Parameters:
        curdt: datetime of current time. Can be a number or pandas series
        origindt: datetime of origin time.

    Returns:
        number of days since origin as a fraction. Returns a numpy or pandas series based on the input.

    Date, Author: --, Krishnakant V. Saboo
    '''
    SECSINDAY = 60*60*24
    return (curdt - origindt).days + (curdt - origindt).seconds/(SECSINDAY)



def tfromstart_days(datdf, datecol='Raw local timestamp', start_date=None):
    '''
    Add column of time since origin in the dataframe. Time is computed as a fraction of days since origin.

    Parameters:
        datdf: pandas dataframe with time information. Must contain column - Raw local timestamp

    Return:
        None. Dataframe datdf is modified inplace to include a column tfromstart_days, which has the fraction of days since origin.

    Date, Author: --, Krishnakant V. Saboo
    '''
    if start_date is None:
        firstrecdate = datetime.datetime.fromisoformat(datdf[datecol].min())
    else:
        firstrecdate = start_date

    datdf['tfromstart_days'] = datdf[datecol].apply(lambda x: daysfrac(datetime.datetime.fromisoformat(x), firstrecdate))
    return




def lsmodel_phase_offset(lsp_object, frequency):
    '''
    Compute phase offset of the LS model at a given frequency.

    Parameters:
        lsp_object: LombScarlge() object fitted to data.
        frequency: period/frequency to sample from the LS model.

    Return:
        phase_offset: phase offset of the fitted model in [0, 2pi)

    Date, Author: --, Krishnakant V. Saboo
    '''
    cons, Acos_phi, Asin_phi = lsp_object.model_parameters(frequency)

    if (Acos_phi >= 0) & (Asin_phi >= 0):
        return np.arctan(Asin_phi/Acos_phi)
    elif (Acos_phi < 0) & (Asin_phi >= 0):
        return np.pi + np.arctan(Asin_phi/Acos_phi)
    elif (Acos_phi < 0) & (Asin_phi < 0):
        return np.pi + np.arctan(Asin_phi/Acos_phi)
    elif (Acos_phi >= 0) & (Asin_phi < 0):
        return 2*np.pi + np.arctan(Asin_phi/Acos_phi)
    else:
        print('Impossible case.')

def lsmodel_phase_offset_peak(theta):
    '''
    Compute phase at which LS model's peak occurs based on the model's phase offset.

    Parameters:
        theta: phase offset (in radians) of LS model.

    Return:
        peak phase: phase at which peak occurs i.e., sin(phase offset + peak phase) = 1.

    Date, Author: --, Krishnakant V. Saboo
    '''
    if theta <= np.pi/2:  # if theta is from 0 to pi/2, then peak occurs before any zero crossing
        return np.pi/2 - theta
    else: # else, then at least one 0 crossing occurs before peak
        return 2*np.pi + np.pi/2 - theta




def round_local_from_utc(dfrow):
    round_utc = (dfrow['Round UTC Timestamp']).to_pydatetime()
    raw_utc = datetime.datetime.fromisoformat(dfrow['Raw UTC Timestamp'])
    raw_local = datetime.datetime.fromisoformat(dfrow['Raw Local Timestamp'])

    round_local = raw_local - (raw_utc - round_utc)
    return round_local



def mean_confidence_interval(myarr=None, sample_mean=None, sample_std=None, sample_size=None, confidence_level=0.95):
    
    # mean and std
    if myarr is not None:
        sample_mean = myarr.mean()
        sample_std = myarr.std()
        sample_size = len(myarr)
    elif (sample_mean is None) or (sample_std is None) or (sample_size is None):
        print('Cannot compute. Provide mean, std, sample size.')
        return None
    
    z_critical = stats.norm.ppf( (1 + confidence_level) / 2)
    standard_error = sample_std / np.sqrt(sample_size)
    
    margin_of_error_z = z_critical * standard_error
    lower_bound_z = sample_mean - margin_of_error_z
    upper_bound_z = sample_mean + margin_of_error_z
    
    return sample_mean, lower_bound_z, upper_bound_z


def corr_repeated_measures(data, x, y, groups):

    '''Learn mixed effects model adn return pvalues, R-squared for fixed effect (Nakagawa formula).
    '''

    ##### apply mixed effects model
    model_formula = f"{y} ~ {x}"
    lmxmodel = smf.mixedlm(model_formula, data=data, groups=data[groups])
    lmxresults = lmxmodel.fit()

    #### compute predictions with fixed effects only
    fixed_pred = lmxresults.params[x] * data[x] + lmxresults.params['Intercept']

    #### compute R2 marginal
    fixed_effects_var = np.var(fixed_pred)
    random_effects_var = lmxresults.cov_re.iloc[0,0]
    residual_var = lmxresults.scale

    R2_marginal = fixed_effects_var / (fixed_effects_var + random_effects_var + residual_var)
    corrval = np.sqrt(R2_marginal) * np.sign(lmxresults.params[x])

    return {'corr':corrval, 'pvalue':lmxresults.pvalues[x], 'lmxmodel':lmxresults}


def include_excluded_features_in_effectsize_df(effectsize, full_feature_list):
    
    ## effectsize of excluded variables
    excluded_variables = [item for item in full_feature_list if item not in list(effectsize.index)]

    esexcluded = pd.DataFrame({'coef': len(excluded_variables)*[0], 
                               'lower': len(excluded_variables)*[0], 
                               'upper': len(excluded_variables)*[0], 
                               'xticks': len(excluded_variables)*[0], 
                               'sig': len(excluded_variables)*[-1]}, index=excluded_variables)

    ## concatenate the inclued and excluded variables' effect sizes
    effectsize = pd.concat((effectsize, esexcluded), ignore_index=False, axis=0)

    ## sort rows based on the order in which the feature appears in the feature list
    effectsize['xticks'] = effectsize.apply(lambda x: -1 * full_feature_list.index(x.name), axis=1)

    return effectsize



def format_patid(patient_id):
    return 'NP%d'%(int(patient_id.split('NPE')[1]))


