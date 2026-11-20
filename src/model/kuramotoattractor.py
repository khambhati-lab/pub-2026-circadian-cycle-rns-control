import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.dataproc import ccutils


class KuramotoAttractorModel:
    
    def __init__(self, theta_left_0, theta_right_0, theta_stim_0, omega_0, couplingsdict):
        self.theta_left_0 = theta_left_0
        self.theta_right_0 = theta_right_0
        self.theta_stim_0 = theta_stim_0
        self.omega_0 = omega_0
        self.freq = omega_0/(2*np.pi)
        self.couplings = couplingsdict  # dictionary of coupling constants

        # in-built variables
        self.theta_cday_0 = - 15 * np.pi / 12
        self.theta_cnight_0 = - 3 * np.pi / 12
        self.theta_reference_0 = 0 # reference oscillator to act as a clock
        self.clus_day_start = 9 * np.pi / 12
        self.clus_day_end = 21 * np.pi / 12
        self.clus_tolerance_angle = 0.001

        # oscillator list that maps oscillator number to name - must be consistent with run_model numbering
        self.oscnummap = {0:'left', 1:'right', 2:'stim', 3:'cday', 4:'cnight', 5:'reference'}
        
    
    # def diff_term_from_attractor(self, fccc_side, theta_fccc, theta_reference, theta_cday, theta_cnight):
    #     fccc_peak_phase = ( theta_reference - theta_fccc ) % (2*np.pi)

    #     if (fccc_peak_phase >= self.clus_day_start) & (fccc_peak_phase < self.clus_day_end):
    #         diff_fccc_clus = self.couplings[f'K_{fccc_side}_cday'] * np.sin(theta_cday - theta_fccc)
    #         closest_clus = 0
    #     else:
    #         diff_fccc_clus = self.couplings[f'K_{fccc_side}_cnight'] * np.sin(theta_cnight - theta_fccc)
    #         closest_clus = 1

    #     return diff_fccc_clus, closest_clus

    def diff_term_from_attractor(self, fccc_side, theta_fccc, theta_reference, theta_cday, theta_cnight):
        fccc_peak_phase = ( theta_reference - theta_fccc ) % (2*np.pi)

        if (fccc_peak_phase >= self.clus_day_start - self.clus_tolerance_angle) & (fccc_peak_phase <= self.clus_day_start + self.clus_tolerance_angle):
            diff_fccc_clus = 0  # no effect from either attractor
            closest_clus = 2   # transition boundary

        elif (fccc_peak_phase >= self.clus_day_end - self.clus_tolerance_angle) & (fccc_peak_phase <= self.clus_day_end + self.clus_tolerance_angle):
            diff_fccc_clus = 0  # no effect from either attractor
            closest_clus = 2   # transition boundary

        elif (fccc_peak_phase > self.clus_day_start + self.clus_tolerance_angle) & (fccc_peak_phase < self.clus_day_end - self.clus_tolerance_angle):
            diff_fccc_clus = self.couplings[f'K_{fccc_side}_cday'] * np.sin(theta_cday - theta_fccc)
            closest_clus = 0
        
        else:
            diff_fccc_clus = self.couplings[f'K_{fccc_side}_cnight'] * np.sin(theta_cnight - theta_fccc)
            closest_clus = 1

        return diff_fccc_clus, closest_clus


    def run_model(self, t_jump, num_T):
        
        num_updates = int(num_T / t_jump)
        
        outmat = np.zeros((6, num_updates))
        outmat[0,0] = self.theta_left_0    # left FC-CC
        outmat[1,0] = self.theta_right_0   # right FC-CC
        outmat[2,0] = self.theta_stim_0   # stimulation
        outmat[3,0] = self.theta_cday_0  # day cluster
        outmat[4,0] = self.theta_cnight_0   # night cluster 
        outmat[5,0] = self.theta_reference_0     # reference clock

        clusmat = np.NaN * np.zeros((2, num_updates))  # cluster
        clusmat[0,0] = self.diff_term_from_attractor(fccc_side='left', theta_fccc=self.theta_left_0, 
            theta_reference=self.theta_reference_0, theta_cday=self.theta_cday_0, theta_cnight=self.theta_cnight_0)[1]
        clusmat[1,0] = self.diff_term_from_attractor(fccc_side='right', theta_fccc=self.theta_right_0, 
            theta_reference=self.theta_reference_0, theta_cday=self.theta_cday_0, theta_cnight=self.theta_cnight_0)[1]

        for tt in range(1, num_updates):
            ### start angles
            theta_left = outmat[0, tt-1]
            theta_right = outmat[1, tt-1]
            theta_stim = outmat[2, tt-1]
            theta_cday = outmat[3, tt-1]
            theta_cnight = outmat[4, tt-1]
            theta_reference = outmat[5, tt-1]

            ### difference terms
            # left oscillator
            diff_left = self.omega_0 + self.couplings['K_left_right'] * np.sin(theta_right - theta_left)
            diff_left += self.couplings['K_left_stim'] * np.sin(theta_stim - theta_left)
            diff_attractor_left, clus_left = self.diff_term_from_attractor(fccc_side='left', theta_fccc=theta_left, 
                theta_reference=theta_reference, theta_cday=theta_cday, theta_cnight=theta_cnight)
            diff_left += diff_attractor_left

            # right oscillator
            diff_right = self.omega_0 + self.couplings['K_right_left'] * np.sin(theta_left - theta_right) 
            diff_right += self.couplings['K_right_stim'] * np.sin(theta_stim - theta_right)
            diff_attractor_right, clus_right = self.diff_term_from_attractor(fccc_side='right', theta_fccc=theta_right, 
                theta_reference=theta_reference, theta_cday=theta_cday, theta_cnight=theta_cnight)
            diff_right += diff_attractor_right

            # stimulation
            diff_stim = self.omega_0

            # cluster and reference oscillators
            diff_cday = self.omega_0
            diff_cnight = self.omega_0
            diff_reference = self.omega_0

            ### new angles
            outmat[0, tt] = theta_left + t_jump * diff_left
            outmat[1, tt] = theta_right + t_jump * diff_right
            outmat[2, tt] = theta_stim + t_jump * diff_stim
            outmat[3, tt] = theta_cday + t_jump * diff_cday
            outmat[4, tt] = theta_cnight + t_jump * diff_cnight
            outmat[5, tt] = theta_reference + t_jump * diff_reference

            # cluster of each oscillator
            clusmat[0, tt] = clus_left
            clusmat[1, tt] = clus_right

        # make all the oscillators lie between 0 and 2pi
        outmat = outmat % (2 * np.pi)

        return outmat, clusmat


    def steady_state_termval(self, current_phases):

        # named phase variables
        theta_left = current_phases[0]
        theta_right = current_phases[1]
        theta_stim = current_phases[2]
        theta_cday = current_phases[3]
        theta_cnight = current_phases[4]
        theta_reference = current_phases[5]

        # left oscillator
        term_left = self.couplings['K_left_right'] * np.sin(theta_right - theta_left)
        term_left += self.couplings['K_left_stim'] * np.sin(theta_stim - theta_left)
        diff_attractor_left, ignore = self.diff_term_from_attractor(fccc_side='left', theta_fccc=theta_left, 
            theta_reference=theta_reference, theta_cday=theta_cday, theta_cnight=theta_cnight)
        term_left += diff_attractor_left

        # right oscillator
        term_right = self.couplings['K_right_left'] * np.sin(theta_left - theta_right) 
        term_right += self.couplings['K_right_stim'] * np.sin(theta_stim - theta_right)
        diff_attractor_right, ignore = self.diff_term_from_attractor(fccc_side='right', theta_fccc=theta_right, 
            theta_reference=theta_reference, theta_cday=theta_cday, theta_cnight=theta_cnight)
        term_right += diff_attractor_right
        
        return term_left, term_right

	
    def oscillator_peak_relphase(self, outmat, t_jump):
        oscpeakdf = [] # store
        peakmat = np.cos(outmat)

        for ii in range(peakmat.shape[0]):
            peak_times, _ = find_peaks(peakmat[ii,:])
            peak_angles = 2 * np.pi * ( ( peak_times * t_jump / self.freq ) % 1 ) # normalized to 1
            
            # osc-peak dataframe
            peroscdf = pd.DataFrame({'osc': self.oscnummap[ii], 'peaknum': np.arange(len(peak_angles)), 
                'peakphase': peak_angles}) 

            # store
            oscpeakdf.append(peroscdf)

        # convert to dataframe
        oscpeakdf = pd.concat(oscpeakdf, ignore_index=True, axis=0)

        return oscpeakdf


def fc_similarity_score(gt_theta_left_final, theta_left_final, gt_theta_right_final, theta_right_final):
    fc_sim = np.cos(gt_theta_left_final - theta_left_final) + np.cos(gt_theta_right_final - theta_right_final)
    fc_sim = ( fc_sim + 2 ) / 4   # normalize to be between [0,1]
    return fc_sim


def fc_similarity_hour(row):
    return ccutils.angle_gap(row, 'gt_left_hour_final', 'left_hour_final') + ccutils.angle_gap(row, 'gt_right_hour_final', 'right_hour_final')



def coupling_symmetric_solution(row, scale=0.1, stfac_left=1, stfac_right=1):
    
    # compute sine of differences
    sin_right_minus_left = np.sin( row['right_peakphase'] - row['left_peakphase'] )
    sin_stim_minus_left = np.sin( row['stim_peakphase'] - row['left_peakphase'] )
    sin_clus_minus_left = np.sin( row['clus_peakphase_left'] - row['left_peakphase'] ) 

    sin_left_minus_right = np.sin( row['left_peakphase'] - row['right_peakphase'] )
    sin_stim_minus_right = np.sin( row['stim_peakphase'] - row['right_peakphase'] )
    sin_clus_minus_right = np.sin( row['clus_peakphase_right'] - row['right_peakphase'] )

    # compute linear system matrix and output column vector
    system_matrix = np.array([[1.0, 1.0, 1.0], 
                              [sin_right_minus_left, stfac_left * sin_stim_minus_left, sin_clus_minus_left], 
                              [sin_left_minus_right, stfac_right * sin_stim_minus_right, sin_clus_minus_right]])
    output_vector = np.array([scale, 0.0, 0.0]) 

    # solve the linear system to obtain coupling constants
    coupling_vector = np.linalg.solve(system_matrix, output_vector)

    # dictionary of coupling constants
    K_brain = coupling_vector[0]
    K_stim = coupling_vector[1]
    K_cluster = coupling_vector[2]

    coupling_dict = {'K_left_right': K_brain, 'K_left_stim': stfac_left * K_stim, 
                     'K_left_cday': K_cluster, 'K_left_cnight': K_cluster, 
                     'K_right_left': K_brain, 'K_right_stim': stfac_right * K_stim,
                     'K_right_cday': K_cluster, 'K_right_cnight': K_cluster}
    
    return coupling_dict



def coupling_illposed_solution(row, scale=0.1):
    
    # compute sine of differences
    sin_right_minus_left = np.sin( row['right_peakphase'] - row['left_peakphase'] )
    sin_stim_minus_left = np.sin( row['stim_peakphase'] - row['left_peakphase'] )
    sin_clus_minus_left = np.sin( row['clus_peakphase_left'] - row['left_peakphase'] ) 

    sin_left_minus_right = np.sin( row['left_peakphase'] - row['right_peakphase'] )
    sin_stim_minus_right = np.sin( row['stim_peakphase'] - row['right_peakphase'] )
    sin_clus_minus_right = np.sin( row['clus_peakphase_right'] - row['right_peakphase'] )

    # compute linear system matrix and output column vector
    system_matrix = np.array([[sin_right_minus_left, sin_stim_minus_left, 0.0], 
                              [sin_left_minus_right, 0.0, sin_stim_minus_right]])
    output_vector = np.array([-scale * sin_clus_minus_left, -scale * sin_clus_minus_right]) 

    # solve the linear system to obtain coupling constants
    coupling_vector, residuals, rank, singular_values = np.linalg.lstsq(system_matrix, output_vector, rcond=None)

    # dictionary of coupling constants
    K_brain = coupling_vector[0]
    K_stim_left = coupling_vector[1]
    K_stim_right = coupling_vector[2]

    coupling_dict = {'K_left_right': K_brain, 'K_left_stim': K_stim_left, 
                     'K_left_cday': scale, 'K_left_cnight': scale, 
                     'K_right_left': K_brain, 'K_right_stim': K_stim_right,
                     'K_right_cday': scale, 'K_right_cnight': scale}
    
    return coupling_dict


def coupling_KclusKbrainOnly_solution(row, scale=0.1):
    
    # compute sine of differences
    sin_right_minus_left = np.sin( row['right_peakphase'] - row['left_peakphase'] )
    sin_clus_minus_left = np.sin( row['clus_peakphase_left'] - row['left_peakphase'] ) 

    sin_left_minus_right = np.sin( row['left_peakphase'] - row['right_peakphase'] )
    sin_clus_minus_right = np.sin( row['clus_peakphase_right'] - row['right_peakphase'] )

    # compute linear system matrix and output column vector
    system_matrix = np.array([[sin_right_minus_left], 
                              [sin_left_minus_right]])
    output_vector = np.array([-scale * sin_clus_minus_left, -scale * sin_clus_minus_right]) 

    # solve the linear system to obtain coupling constants
    coupling_vector, residuals, rank, singular_values = np.linalg.lstsq(system_matrix, output_vector, rcond=None)

    # dictionary of coupling constants
    K_brain = coupling_vector[0]

    coupling_dict = {'K_left_right': K_brain, 'K_left_stim': 0.0, 
                     'K_left_cday': scale, 'K_left_cnight': scale, 
                     'K_right_left': K_brain, 'K_right_stim': 0.0,
                     'K_right_cday': scale, 'K_right_cnight': scale}
    
    return coupling_dict


def coupling_KclusOnly_solution(row, scale=0.1):
    
    # no fitting, just return the scale
    coupling_dict = {'K_left_right': 0.0, 'K_left_stim': 0.0, 
                     'K_left_cday': scale, 'K_left_cnight': scale, 
                     'K_right_left': 0.0, 'K_right_stim': 0.0,
                     'K_right_cday': scale, 'K_right_cnight': scale}
    
    return coupling_dict


def coupling_nobrainintc_solution(row, scale=0.1, left_fac=1, right_fac=1, left_is_reference=True):
    
    # compute sine of differences
    sin_stim_minus_left = np.sin( row['stim_peakphase'] - row['left_peakphase'] )
    sin_clus_minus_left = np.sin( row['clus_peakphase_left'] - row['left_peakphase'] ) 

    sin_stim_minus_right = np.sin( row['stim_peakphase'] - row['right_peakphase'] )
    sin_clus_minus_right = np.sin( row['clus_peakphase_right'] - row['right_peakphase'] )

    # compute linear system matrix and output column vector
    if left_is_reference:
        system_matrix = np.array([[1.0, 1.0, 0.0], 
                                  [sin_stim_minus_left, sin_clus_minus_left, 0.0], 
                                  [0.0, sin_clus_minus_right, sin_stim_minus_right]])
        output_vector = np.array([scale, 0.0, 0.0]) 

    else:
        system_matrix = np.array([[0.0, 1.0, 1.0], 
                                  [sin_stim_minus_left, sin_clus_minus_left, 0.0], 
                                  [0.0, sin_clus_minus_right, sin_stim_minus_right]])
        output_vector = np.array([scale, 0.0, 0.0]) 


    ## solve the linear system to obtain coupling constants
    # coupling_vector = np.linalg.solve(system_matrix, output_vector)
    coupling_vector, residuals, rank, singular_values = np.linalg.lstsq(system_matrix, output_vector, rcond=None)

    # dictionary of coupling constants
    K_left_stim = coupling_vector[0]
    K_cluster = coupling_vector[1]
    K_right_stim = coupling_vector[2]

    coupling_dict = {'K_left_right': 0.0, 'K_left_stim': K_left_stim, 
                     'K_left_cday': K_cluster, 'K_left_cnight': K_cluster, 
                     'K_right_left': 0.0, 'K_right_stim': K_right_stim,
                     'K_right_cday': K_cluster, 'K_right_cnight': K_cluster}
    
    return coupling_dict



def test_couplings_with_forward_model(theta_init_dict, coupling_dict, t_jump=0.001, num_T=180):

    class_input = theta_init_dict | {'couplingsdict': coupling_dict}
    kattmodel = KuramotoAttractorModel(**class_input)

    # run the model forward
    outmat, clusmat = kattmodel.run_model(t_jump, num_T)

    # peak phase of oscillators relative to the fixed clock
    osppdf = kattmodel.oscillator_peak_relphase(outmat, t_jump)

    # check steady state term
    ssterm_left, ssterm_right = kattmodel.steady_state_termval(outmat[:,-1])

    return osppdf, outmat, clusmat, np.abs(ssterm_left) + np.abs(ssterm_right)




class AlternateStimulation:
    
    def __init__(self, datrow, klrmodel, kstimmodel, aucintdict, auccondict, klrscaling=2.718, kstimscaling=2.358):
        self.data = datrow
        self.klrmodel = klrmodel
        self.kstimmodel = kstimmodel
        self.aucintdict = aucintdict
        self.auccondict = auccondict
        self.klrscaling = klrscaling
        self.kstimscaling = kstimscaling
        
      
    def get_km_init(self, stimhour):
        """Initial phase of the oscillators for Kuramoto model with attractors"""

        # initial phases of the oscillators
        init_thetas = -1 * self.data[['left_peakphase_prev', 'right_peakphase_prev']] # negative sign so that the peak occurs at the correct phase
        init_thetas['alt_stimpeaktime'] = -np.pi/12 * stimhour

        init_thetas.rename(index={'left_peakphase_prev':'theta_left_0', 
                                  'right_peakphase_prev':'theta_right_0', 
                                  'alt_stimpeaktime':'theta_stim_0'}, 
                           inplace=True)

        init_thetas =  init_thetas.to_dict()  | {'omega_0':2*np.pi}
        
        return init_thetas


    def fitted_coupling(self, stimhour):
        """Return true fitted coupling constants. This is just to test the forward model code."""
        modified_data = self.data.copy()

        return modified_data[[item for item in modified_data.index if 'K_' in item]].to_dict()

    
    def coupling_for_given_stim(self, stimhour):
        """Predicted Kuramoto coupling constants for given stimulation peak hour."""

        modified_data = self.data.copy() # copy of the data to be modified with new stim value

        ##### specify peak hour and compute predicted stimulation AUC values  #####
        sin_stimhour = np.sin(np.pi / 12 * stimhour)
        cos_stimhour = np.cos(np.pi / 12 * stimhour)
        aucbetween = self.aucintdict[stimhour]  # expected AUC interval at given hour
        aucoutside = self.auccondict[stimhour]  # expected AUC converse at given hour

        modified_data['sin_stim'] = sin_stimhour
        modified_data['cos_stim'] = cos_stimhour
        modified_data['stimaucbetween'] = aucbetween
        modified_data['stimaucoutside'] = aucoutside

        #### obtain coupling constants for given initial FC-CC features and stimulation #####

        # K left and right
        K_left_right_pred = self.klrmodel.test_linreg_model(modified_data).item()
        K_left_right_pred = self.klrscaling * K_left_right_pred

        # K left and stimulation
        cycle_left_cols = ['left_amp_prev', 'sin_left_prev', 'cos_left_prev'] 
        modified_data_left = modified_data.copy() # rename columns to be consistent with features for hierarchical model
        modified_data_left.rename(dict(zip(cycle_left_cols, [item.replace('left','cycle') for item in cycle_left_cols])), inplace=True)
        K_left_stim_pred = self.kstimmodel.test_linreg_model(modified_data_left.to_frame().T).item()
        K_left_stim_pred = self.kstimscaling * K_left_stim_pred

        # K right and stimulation
        cycle_right_cols = ['right_amp_prev', 'sin_right_prev', 'cos_right_prev']
        modified_data_right = modified_data.copy() # rename columns to be consistent with features for hierarchical model
        modified_data_right.rename(dict(zip(cycle_right_cols, [item.replace('right','cycle') for item in cycle_right_cols])), inplace=True)
        K_right_stim_pred = self.kstimmodel.test_linreg_model(modified_data_right.to_frame().T).item()
        K_right_stim_pred = self.kstimscaling * K_right_stim_pred

        # K attractor
        K_attractor_pred = modified_data['K_left_cday']

        #### coupling constant values stored as a dictionary
        coupling_dict = {'K_left_right': K_left_right_pred, 'K_left_stim': K_left_stim_pred, 
                         'K_left_cday': K_attractor_pred, 'K_left_cnight': K_attractor_pred, 
                         'K_right_left': K_left_right_pred, 'K_right_stim': K_right_stim_pred,
                         'K_right_cday': K_attractor_pred, 'K_right_cnight': K_attractor_pred}
            
        return coupling_dict

        
    def kuramoto_sol_altstim(self, stimhour, scaling=1):
        """Kuramoto model forward solution for given peak hour of stimulation."""

        init_thetas = self.get_km_init(stimhour)  # initial value of the oscillators
        predicted_couplings = self.coupling_for_given_stim(stimhour)   # predicted coupling constant for given stimulation peak hour
        if scaling != 1:
            predicted_couplings = ( pd.Series(predicted_couplings) / scaling ).to_dict()

        # run model forward with fitted coupling constants
        osppdf, ig1, ig2, ssterm = test_couplings_with_forward_model(theta_init_dict=init_thetas, 
                                                                     coupling_dict=predicted_couplings)

        # output peak sync
        peakphase_left_final = osppdf.loc[osppdf.loc[(osppdf['osc']=='left')]['peaknum'].idxmax()]['peakphase']
        peakphase_right_final = osppdf.loc[osppdf.loc[(osppdf['osc']=='right')]['peaknum'].idxmax()]['peakphase']
        alt_peaksync = ccutils.peak_sync_from_angles(peakphase_left_final, peakphase_right_final)
        
        return init_thetas | predicted_couplings, alt_peaksync, osppdf, ssterm 


