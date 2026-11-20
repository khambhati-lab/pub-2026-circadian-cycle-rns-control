import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.model import kuramotoattractor


def load_model_datadf(datapath):

	## load data
	datadf = pd.read_csv(datapath)

	## rename columns
	rename_dict = {'cycle1amp_local':'left_amp', 'cycle1peak_local':'left_peaktime', 
				'cycle2amp_local':'right_amp', 'cycle2peak_local':'right_peaktime', 
				'cycle1amp_local_prev':'left_amp_prev', 'cycle1peak_local_prev':'left_peaktime_prev', 
				'cycle2amp_local_prev':'right_amp_prev', 'cycle2peak_local_prev':'right_peaktime_prev', 
				'max_hour':'stim_peaktime', 'auc':'stimauc', 'auc_converse':'stimaucoutside', 
				'auc_interval':'stimaucbetween'}

	datadf.rename(columns=rename_dict, inplace=True)

	## columns relevant to the model
	moddatadf = datadf[['npecode', 'dur_day',
						'left_amp_prev', 'left_peaktime_prev', 'right_amp_prev', 'right_peaktime_prev', 
						'stim_peaktime', 'stimauc', 'stimaucoutside', 'stimaucbetween', 
						'left_amp', 'left_peaktime', 'right_amp', 'right_peaktime']].copy()

	## model uses the hour in phase (radians). convert time angles
	for col in ['left_peaktime_prev', 'right_peaktime_prev', 'left_peaktime', 'right_peaktime', 'stim_peaktime']:
		newcol = col.replace('time', 'phase')
		moddatadf[newcol] = np.pi/12 * moddatadf[col]

	dayclus_peakphase = 15 * np.pi / 12
	nightclus_peakphase = 3 * np.pi / 12

	## cluster assigment
	moddatadf['clus_peakphase_left'] = moddatadf['left_peaktime'].apply(lambda x: dayclus_peakphase if (x >= 9) and (x < 21) else nightclus_peakphase)
	moddatadf['clus_peakphase_right'] = moddatadf['right_peaktime'].apply(lambda x: dayclus_peakphase if (x >= 9) and (x < 21) else nightclus_peakphase)

	return moddatadf



def fit_coupling_for_sample(df_index, row, fit_coupling_func, coupling_scale_list, converg_ssterm_thresh=1e-5):
	
	# initial and final peak hours ground truth
	init_thetas = -1 * row[['left_peakphase_prev', 'right_peakphase_prev', 'stim_peakphase']] # negative sign so that the peak occurs at the correct phase
	init_thetas.rename(index={'left_peakphase_prev':'theta_left_0', 'right_peakphase_prev':'theta_right_0', 'stim_peakphase':'theta_stim_0'}, 
		inplace=True)
	init_thetas = init_thetas.to_dict() | {'omega_0':2*np.pi}

	gt_theta_left_final = row['left_peakphase']
	gt_theta_right_final = row['right_peakphase']

	fc_sim_best = -10   # minimum similarity score
	best_coups = {}
	best_scale = np.NaN
	best_ssterm = np.NaN
	best_theta_left_final, best_theta_right_final = np.NaN, np.NaN

	for coupscale in coupling_scale_list:
		#### compute coupling constants
		## fit coupling constants        
		temp_coups = fit_coupling_func(row, scale=coupscale)

		# run model forward with fitted coupling constants
		osppdf, outmat, clusmat, ssterm = kuramotoattractor.test_couplings_with_forward_model(theta_init_dict=init_thetas, 
																								coupling_dict=temp_coups)

		# compare predicted phase and ground truth phase; compute similarity [-2,2]
		theta_left_final = osppdf.loc[osppdf.loc[(osppdf['osc']=='left')]['peaknum'].idxmax()]['peakphase'].item()
		theta_right_final = osppdf.loc[osppdf.loc[(osppdf['osc']=='right')]['peaknum'].idxmax()]['peakphase'].item()

		fc_sim = kuramotoattractor.fc_similarity_score(gt_theta_left_final, theta_left_final, gt_theta_right_final, theta_right_final)

		if (ssterm < converg_ssterm_thresh) & (fc_sim > fc_sim_best): # if similarity is better than best seen so far, update the parameters
			fc_sim_best = fc_sim
			best_coups = temp_coups.copy()
			best_scale = coupscale
			best_ssterm = ssterm
			best_theta_left_final = theta_left_final
			best_theta_right_final = theta_right_final
			if fc_sim_best >= (1 - 5e-3): # if close to best is achieved, then exit the loop
				break

	# store the coupling constants for given set
	best_scale_dict = {'index': df_index, 'scale': best_scale, 'fc_sim': fc_sim_best, 'steadystateterm':best_ssterm}
	best_finalphase_dict = {'gt_left_hour_init':row['left_peaktime_prev'], 'gt_left_hour_final':12/np.pi * gt_theta_left_final, 'left_hour_final':12/np.pi * best_theta_left_final, 
							'gt_right_hour_init':row['right_peaktime_prev'], 'gt_right_hour_final':12/np.pi * gt_theta_right_final, 'right_hour_final':12/np.pi * best_theta_right_final}

	# return 
	return best_scale_dict | best_coups | best_finalphase_dict


def fit_coupling_for_population(moddatadf, estimator, coupling_scale_list, verbose=True):

	## pick function based on the estimator to be used
	if estimator == 'attractor':
		fit_coupling_func = kuramotoattractor.coupling_KclusOnly_solution
	
	elif estimator == 'attractor_brain':
		fit_coupling_func = kuramotoattractor.coupling_KclusKbrainOnly_solution
	
	elif estimator == 'attractor_brain_stim':
		fit_coupling_func = kuramotoattractor.coupling_illposed_solution

	else:
		raise ValueError(f"Unknown estimator: {estimator}.")


	coupdf = []  # best fitted coupling constants

	for df_index, row in moddatadf.iterrows():

		# coupling for one sample
		samp_dict = fit_coupling_for_sample(df_index, row, fit_coupling_func, coupling_scale_list)
		coupdf.append( samp_dict ) # store in list

		if verbose and (df_index % 10 == 0): print(f'indices processed: {df_index}')

	# convert to dataframe
	coupdf = pd.DataFrame(coupdf)
	coupdf.set_index('index', inplace=True)

	return coupdf



if __name__ == "__main__":

	## path
	dirpath = 'data/processed/kuramotoattractor/'
	datapath = dirpath + 'paired_fcccprev_fccc_stim.csv'

	## dataframe for modelling
	moddatadf = load_model_datadf(datapath)

	## estimator and the coupling scales to try
	estimator_scale_dict = {'attractor': [2], 'attractor_brain': [15], 'attractor_brain_stim': [2]}

	## iterate over cases and fit coulings
	for key, val in estimator_scale_dict.items():

		print(f'!!!!! Starting estimation for {key} model !!!!!')

		# fit coupling constants
		coupdf = fit_coupling_for_population(moddatadf, key, val)  # fit coupling constants

		# save the coupling constant along with the model data
		savedf = pd.concat((moddatadf, coupdf), axis=1, ignore_index=False)
		save_filename = dirpath + f'fitted_couplings_{key}_24Sep2025.csv'
		savedf.to_csv(save_filename, index=True, index_label='index')






	