import numpy as np
import pandas as pd
import statsmodels.regression as sm
from src.dataproc import ccutils
from copy import deepcopy


def cross_validation(regdf, regfeat_cols, yvar, cv_folds_list, cv_var, remove_outliers=True, verbose=True):

	# handle outliers
	df_index = ~regdf[cv_var].isna()   # simple condition that I know is true for all indices
	
	if remove_outliers:

		for featcol in regfeat_cols + [yvar]:
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
		
		# include constant in the features
		features_x['const'] = 1
		
		# train the model
		linmodel = sm.linear_model.OLS(endog=out_y, exog=features_x, hasconst=True)
		linmodel_results = linmodel.fit()

		## testing data
		# get test data and normalize it using training data
		test_features_x = test_df[regfeat_cols].copy()
		test_features_x = ( test_features_x - features_x_mean ) / features_x_std
		test_features_x['const'] = 1
		test_out_y = test_df[yvar].copy()

		# apply model to test data
		test_pred = linmodel_results.predict(exog=test_features_x, transform=False)

		# baseline predict (mean of the training data)
		base_pred = out_y.mean()

		# store results
		num_test = len(test_pred)
		fold_resdf = pd.DataFrame({
			'index': test_out_y.index,
			cv_var: num_test*[test_fold], 
			'gt': test_out_y.values, 
			'modpred': test_pred.values, 
			'basemean': num_test*[base_pred]})
		fold_resdf.set_index('index', inplace=True)

		resdf.append(fold_resdf)
		moddict[test_fold] = deepcopy(linmodel_results)


	# convert to dataframe
	resdf = pd.concat(resdf, ignore_index=False, axis=0)

	if yvar == 'sz_red':
		# clip predictions to 0 / 1
		resdf['modpred'] = np.clip(a=resdf['modpred'], a_max=1, a_min=0)
		if verbose: print('Clipped output between 0-1.')

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

		# model p-values
		sr1 = pd.Series({'fvalue': trmod.f_pvalue})

		# coefficients
		sr2 = trmod.params.copy()
		cols = sr2.index
		sr2.rename(dict(zip(cols, [item+'__coef' for item in cols])), inplace=True)

		# coefficients pvalues
		sr3 = trmod.pvalues.copy()
		sr3.rename(dict(zip(cols, [item+'__pvalue' for item in cols])), inplace=True)

		# confidence interval lower limit
		sr4 = trmod.conf_int()[0].copy()
		sr4.rename(dict(zip(cols, [item+'__lower' for item in cols])), inplace=True) 

		# confidence interval upper limit
		sr5 = trmod.conf_int()[1].copy()
		sr5.rename(dict(zip(cols, [item+'__upper' for item in cols])), inplace=True) 

		# concenate the series
		modsumdf.append(pd.concat((sr0, sr1, sr2, sr3, sr4, sr5)))

	# concate information from all folds
	modsumdf = pd.concat(modsumdf, axis=1, ignore_index=True).T

	return modsumdf


def combined_llf(moddict):
	return np.mean([value.llf for key, value in moddict.items()])


def recursive_feature_elimination(regdf, full_feature_list, yvar, cv_folds_list, cv_var, remove_outliers, verbose=True):

	alpha_level = 0.05
	# perform cross-validation with the feature list
	ignore, linmoddict = cross_validation(regdf=regdf, regfeat_cols=full_feature_list, yvar=yvar, 
		cv_folds_list=cv_folds_list, cv_var=cv_var, remove_outliers=remove_outliers, verbose=verbose)

	# compute the number of times each feature is significant. Ignore the constant feature.
	lmsummary = summarize_model_cv(linmoddict, cv_var=cv_var)
	pvalue_cols = [item for item in lmsummary.columns if ('__pvalue' in item) and ('const' not in item)]
	sigcounts = (lmsummary[pvalue_cols] < alpha_level).sum()

	if verbose: print('Significance counts of each feature:\n', sigcounts)

	# find the features which are significant at least in one CV fold
	updated_feature_list = [item.split('__pvalue')[0] for item in list(sigcounts.index[sigcounts > 0])]

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


def effect_size_across_folds(lmsummary):
	coef_cols = [item for item in lmsummary.columns if ('__coef' in item) and ('const' not in item)]
	lower_cols = [item for item in lmsummary.columns if ('__lower' in item) and ('const' not in item)]
	upper_cols = [item for item in lmsummary.columns if ('__upper' in item) and ('const' not in item)]

	effectsize = lmsummary[coef_cols + lower_cols + upper_cols].mean()
	effectsize = effectsize.reset_index()
	effectsize.rename(columns={'index':'feature_type', 0:'value'}, inplace=True)
	effectsize['feature'] = effectsize['feature_type'].apply(lambda x: x.split('__')[0])
	effectsize['type'] = effectsize['feature_type'].apply(lambda x: x.split('__')[1])
	effectsize = effectsize.pivot_table(index=['feature'], columns=['type'], values=['value'], sort=False)
	effectsize.columns = effectsize.columns.to_flat_index()
	effectsize.rename(columns=dict(zip(effectsize.columns, [item[1] for item in effectsize.columns])), inplace=True)
	effectsize['xticks'] = -np.arange(effectsize.shape[0])
	effectsize['sig'] = effectsize.apply(lambda x: np.sign(x['lower']) * np.sign(x['upper']), axis=1)

	return effectsize



class LinRegModel():

	def __init__(self, regdf, regfeat_cols, yvar, cv_var, test_fold, remove_outliers):
		
		self.regdf = regdf
		self.regfeat_cols = regfeat_cols
		self.yvar = yvar
		self.cv_var = cv_var
		self.test_fold = test_fold
		self.remove_outliers = remove_outliers


	def linreg_data_outliers_and_split(self, verbose=False):
		"""Remove outliers and split linear regression data"""

		# get dataset
		regdf = self.regdf.copy()

		# handle outliers
		df_index = ~regdf[self.cv_var].isna()   # simple condition that I know is true for all indices
		
		if self.remove_outliers:

			for featcol in self.regfeat_cols + [self.yvar]:
				feat_out_thresh = ccutils.outlier_threshold(regdf[featcol])
				df_index = (df_index & (regdf[featcol].between(*feat_out_thresh)) )

			if verbose: print(f'Removed outliers. Num entries: {df_index.sum()}.')

		else:
			if verbose: print(f'Outliers NOT removed. Num entries: {df_index.sum()}.')

		# dataframe after removing outliers
		fulldf = regdf.loc[df_index].copy()  

		# split into training and test
		train_df = fulldf.loc[fulldf[self.cv_var] != self.test_fold].copy()
		test_df = fulldf.loc[fulldf[self.cv_var] == self.test_fold].copy()

		return train_df, test_df



	def train_linreg_model(self, train_df):
		"""Get trained linear regression model along with normalization parameters"""

		# get features and outcome
		features_x = train_df[self.regfeat_cols].copy()
		out_y = train_df[self.yvar].copy()

		# normalization
		lr_features_x_mean = features_x.mean()
		lr_features_x_std = features_x.std()
		features_x = ( features_x - lr_features_x_mean ) / lr_features_x_std
		
		# include constant in the features
		features_x['const'] = 1
		
		# train the model
		linmodel = sm.linear_model.OLS(endog=out_y, exog=features_x, hasconst=True)
		linmodel_results = linmodel.fit()

		return linmodel_results, lr_features_x_mean, lr_features_x_std


	def test_linreg_model(self, test_df):
		"""Apply trained linear regression to test data"""

		## testing data
		# get test data and normalize it using training data
		test_features_x = test_df[self.regfeat_cols].copy()
		test_features_x = ( test_features_x - self.lr_norm_mean ) / self.lr_norm_std
		test_features_x['const'] = 1
		
		# apply model to test data
		test_pred = self.linmodel.predict(exog=test_features_x, transform=False)

		return test_pred



	def main(self):
		"""Main function that returns the test set results"""

		# split the data into training and test set
		train_df, test_df = self.linreg_data_outliers_and_split()

		# train linear regression model
		self.linmodel, self.lr_norm_mean, self.lr_norm_std = self.train_linreg_model(train_df)

		if len(test_df) > 0:  # if test data is available

			# test linear regression model
			test_out_y = test_df[self.yvar].copy()
			test_pred = self.test_linreg_model(test_df)

			# store results
			num_test = len(test_pred)
			lr_resdf = pd.DataFrame({
				'index': test_out_y.index,
				self.cv_var: num_test*[self.test_fold], 
				'gt': test_out_y.values, 
				'modpred': test_pred.values})
			lr_resdf.set_index('index', inplace=True)

		else:
			print(f'{self.test_fold}: No LR test samples')
			lr_resdf = pd.DataFrame({})


		return lr_resdf
