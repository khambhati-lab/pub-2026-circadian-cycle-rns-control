import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression	
from sklearn.svm import LinearSVC
from src.dataproc import ccutils
from copy import deepcopy


def cross_validation(regdf, regfeat_cols, yvar, cv_folds_list, cv_var, remove_outliers=True, verbose=True):

	# handle outliers
	df_index = ~regdf[cv_var].isna()   # simple condition that I know is true for all indices
	
	if remove_outliers:

		for featcol in regfeat_cols:
			feat_out_thresh = ccutils.outlier_threshold(regdf[featcol])
			df_index = (df_index & (regdf[featcol].between(*feat_out_thresh)) )

		if verbose: print(f'Removed outliers. Num entries: {df_index.sum()}.')

	else:
		if verbose: print(f'Outliers NOT removed. Num entries: {df_index.sum()}.')

	# dataframe after removing outliers
	fulldf = regdf.loc[df_index].copy()  

	cv_folds_list_after_outliers = fulldf[cv_var].unique()

	# variables to store results
	resdf = [] 
	moddict = {}

	# cross validation
	for test_fold in cv_folds_list:

		if test_fold not in cv_folds_list_after_outliers:
			continue

		# split into training and test
		train_df = fulldf.loc[fulldf[cv_var] != test_fold].copy()
		test_df = fulldf.loc[fulldf[cv_var] == test_fold].copy()

		# get features and outcome
		features_x = train_df[regfeat_cols].copy()
		out_y = train_df[yvar].copy()

		# normalization
		features_x_mean = features_x.mean()
		features_x_std = features_x.std()
		features_x = ( features_x - features_x_mean ) / features_x_std
				
		# train the model
		# rfmodel = RandomForestClassifier(n_estimators=15)
		rfmodel = LogisticRegression(penalty='l2')
		# rfmodel = LinearSVC(penalty='l2', class_weight='balanced')
		rfmodel.fit(X=features_x, y=out_y)

		## testing data
		# get test data and normalize it using training data
		test_features_x = test_df[regfeat_cols].copy()
		test_features_x = ( test_features_x - features_x_mean ) / features_x_std
		test_out_y = test_df[yvar].copy()

		# apply model to test data
		# test_pred = rfmodel.predict(X=test_features_x)
		test_pred = rfmodel.predict_proba(X=test_features_x)[:,1] # proba for class 1

		# baseline predict (mean of the training data)
		base_pred = out_y.mean()

		# store results
		num_test = len(test_pred)
		fold_resdf = pd.DataFrame({
			'index': test_out_y.index,
			cv_var: num_test*[test_fold], 
			'gt': test_out_y.values, 
			'modpred': test_pred, 
			'basemean': num_test*[base_pred]})
		fold_resdf.set_index('index', inplace=True)

		resdf.append(fold_resdf)
		moddict[test_fold] = deepcopy(rfmodel)


	# convert to dataframe
	resdf = pd.concat(resdf, ignore_index=False, axis=0)

	if verbose: print('Cross-validation complete.')

	return resdf, moddict


def foldwise_mse(preddf, cv_var, diff_tuples):

	msedf = []

	for tup in diff_tuples:

		modmsedf = preddf.groupby(by=cv_var).apply(lambda x: ((x[tup[0]] - x[tup[1]]) ** 2).mean() )
		modmsedf.name = tup[2]

		msedf.append(modmsedf)

	# concatenate the results
	msedf = pd.concat(msedf, axis=1, ignore_index=False)
	return msedf


def combined_mse(preddf, diff_tuples):

	msedf = []

	for tup in diff_tuples:

		modmsedf = pd.DataFrame( {tup[2]: [((preddf[tup[0]] - preddf[tup[1]]) ** 2).mean()]} )

		msedf.append(modmsedf)

	# concatenate the results
	msedf = pd.concat(msedf, axis=1, ignore_index=False)
	return msedf



def summarize_model_cv(moddict, cv_var):

	modsumdf = []

	for cv_fold, trmod in moddict.items():

		# fold information
		sr0 = pd.Series({cv_var: cv_fold})

		# feature importance
		# sr1 = pd.Series(dict(zip(trmod.feature_names_in_, trmod.feature_importances_)))
		sr1 = pd.Series(dict(zip(trmod.feature_names_in_, trmod.coef_[0])))

		# concenate the series
		modsumdf.append(pd.concat((sr0, sr1)))

	# concate information from all folds
	modsumdf = pd.concat(modsumdf, axis=1, ignore_index=True).T

	return modsumdf




def recursive_feature_elimination(regdf, full_feature_list, yvar, cv_folds_list, cv_var, remove_outliers, verbose=True):

	fi_threshold = 0.05   # minimum feature importance
	# perform cross-validation with the feature list
	ignore, glmmoddict = cross_validation(regdf=regdf, regfeat_cols=full_feature_list, yvar=yvar, 
		cv_folds_list=cv_folds_list, cv_var=cv_var, remove_outliers=remove_outliers, verbose=verbose)

	# compute the number of times each feature is significant
	glmsummary = summarize_model_cv(glmmoddict, cv_var=cv_var)
	glmsummary.set_index(cv_var, inplace=True)  # index by cv var
	frac_feat_imp = glmsummary.mean().abs() / glmsummary.mean().abs().sum()

	if verbose: print('Fractional feature importance of each feature:\n', frac_feat_imp)

	# find the features which have important more than a feature importance threshold
	updated_feature_list = list(frac_feat_imp.index[frac_feat_imp >= fi_threshold])

	### recursion
	if len(updated_feature_list) == 0: # no features are signficant
		return []  # return empty list

	elif len(updated_feature_list) == len(full_feature_list): # entire list of features is significant
		return full_feature_list  # converged, return the full list

	elif len(updated_feature_list) < len(full_feature_list): # if some features were non-significant
		# run recursion with the updated list of features
		return recursive_feature_elimination(regdf, updated_feature_list, yvar, cv_folds_list, cv_var, remove_outliers, verbose)

	else:
		# this condition should never be reached
		raise ValueError



