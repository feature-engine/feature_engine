import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from target_mean_encoding import TargetMeanEncoding


def test_target_mean_encoding_categorical_variables_r2_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(variables=['pclass', 'sex', 'cabin', 'embarked'],  target='survived', metrics_score='r2_score', variables_type=True)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': {0: 'sex', 1: 'pclass', 2: 'embarked', 3: 'cabin'},
		'Importance': { 0: 0.2629646305571747,
						1: 0.10118121894576682,
						2: 0.04226338459397283,
						3: -0.1563034966687058}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_categorical_variables_roc_auc_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(variables=['pclass', 'sex', 'cabin', 'embarked'], target='survived', metrics_score='roc_auc_score', variables_type=True)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'sex', 1: 'pclass', 2: 'embarked', 3: 'cabin'},
		 'Importance': {  0: 0.7499585199933632,
						  1: 0.6700956805486422,
						  2: 0.5933576682705604,
						  3: 0.4795918367346939}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_numerical_variables_r2_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(variables=['fare', 'age'], 
			                         target='survived', 
			                         metrics_score='r2_score', 
			                         variables_type=False)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': {0: 'fare_binned', 1: 'age_binned'},
			'Importance': {0: 0.12165440084922352, 1: -0.10817844058079684}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_numerical_variables_roc_auc_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(variables=['fare', 'age'], 
			                         target='survived', 
			                         metrics_score='roc_auc_score', 
			                         variables_type=False)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'fare_binned', 1: 'age_binned'},
			'Importance': {0: 0.7070958464686686, 1: 0.5086278413804546}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_r2_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='r2_score', 
			                         variables_type=False)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': { 0: 'fare_binned',
					  1: 'age_binned',
					  2: 'parch_binned',
					  3: 'sibsp_binned',
					  4: 'body_binned'},
		'Importance': {0: 0.12165440084922352,
					  1: -0.10817844058079684,
					  2: -0.2740295209340766,
					  3: -0.2825100602595454,
					  4: -0.655664312229187}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_roc_auc_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='roc_auc_score', 
			                         variables_type=False)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'fare_binned',
						  1: 'parch_binned',
						  2: 'sibsp_binned',
						  3: 'age_binned',
						  4: 'body_binned'},
		'Importance': {   0: 0.7070958464686686,
						  1: 0.6078203639179248,
						  2: 0.5969526021790831,
						  3: 0.5086278413804546,
						  4: 0.4268292682926829}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_categorical_variables_r2_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='r2_score', 
			                         variables_type=True)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': { 0: 'boat',
					  1: 'sex',
					  2: 'pclass',
					  3: 'embarked',
					  4: 'ticket',
					  5: 'home_dest',
					  6: 'cabin',
					  7: 'name'},
		'Importance': {0: 0.8804756896980712,
					  1: 0.2629646305571747,
					  2: 0.10118121894576682,
					  3: 0.04226338459397283,
					  4: -0.12026397415334733,
					  5: -0.12612563609991123,
					  6: -0.1563034966687058,
					  7: -0.5975609756097566}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_categorical_variables_roc_auc_score(df_target_mean_encoding):
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='roc_auc_score', 
			                         variables_type=True)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'boat',
						  1: 'sex',
						  2: 'ticket',
						  3: 'pclass',
						  4: 'home_dest',
						  5: 'embarked',
						  6: 'name',
						  7: 'cabin'},
		'Importance': {   0: 0.9679774348763895,
						  1: 0.7499585199933632,
						  2: 0.6828300425861402,
						  3: 0.6700956805486422,
						  4: 0.6157845251921907,
						  5: 0.5933576682705604,
						  6: 0.5,
						  7: 0.4795918367346939}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_roc_auc_score_quantiles_50(df_target_mean_encoding):
	"""check all numerical variables in roc auc score with a quantiles 50."""
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='roc_auc_score', 
			                         variables_type=False, 
			                         quantiles=50)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'fare_binned',
						  1: 'parch_binned',
						  2: 'sibsp_binned',
						  3: 'age_binned',
						  4: 'body_binned'},
		'Importance': {   0: 0.6945965378021128,
						  1: 0.6149134450528179,
						  2: 0.5990404291798019,
						  3: 0.5312897516730269,
						  4: 0.49390243902439024}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_roc_auc_score_quantiles_100(df_target_mean_encoding):
	"""check all numerical variables in roc auc score with a quantiles 100."""
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='roc_auc_score', 
			                         variables_type=False, 
			                         quantiles=100)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'roc_auc_score': {0: 'fare_binned',
						  1: 'parch_binned',
						  2: 'sibsp_binned',
						  3: 'age_binned',
						  4: 'body_binned'},
		'Importance': {   0: 0.7046485260770975,
						  1: 0.6128532713898567,
						  2: 0.5990404291798019,
						  3: 0.5489049278247885,
						  4: 0.5}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_r2_score_quantiles_50(df_target_mean_encoding):
	"""check all numerical variables in r2 score with a quantiles 50."""
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='r2_score', 
			                         variables_type=False, 
			                         quantiles=50)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': { 0: 'fare_binned',
					  1: 'age_binned',
					  2: 'parch_binned',
					  3: 'sibsp_binned',
					  4: 'body_binned'},
		'Importance': {0: 0.10929478380289304,
					  1: -0.119307402816349,
					  2: -0.26828212024409837,
					  3: -0.27403215618581944,
					  4: -0.6032913974189684}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_target_mean_encoding_all_numerical_variables_r2_score_quantiles_100(df_target_mean_encoding):
	"""check all numerical variables in r2 score with a quantiles 100."""
	transformer = TargetMeanEncoding(target='survived', 
			                         metrics_score='r2_score', 
			                         variables_type=False, 
			                         quantiles=100)
	X = transformer.fit_transform(df_target_mean_encoding)

	# expected result
	df = pd.DataFrame({
		'r2_score': { 0: 'fare_binned',
					  1: 'age_binned',
					  2: 'parch_binned',
					  3: 'sibsp_binned',
					  4: 'body_binned'},
		'Importance': {0: 0.10335140205109206,
					  1: -0.13336816575113608,
					  2: -0.2738609046033704,
					  3: -0.27403215618581944,
					  4: -0.5975609756097566}
		}
	)
	pd.testing.assert_frame_equal(X, df)


def test_error_if_fit_input_not_dataframe(self):
	"""check if input is not a dataframe."""
	with pytest.raises(TypeError):
	    TargetMeanEncoding(target='survived', 
	                       variables_type=True)().fit({"Name": ["Karthik"]})


def test_not_fitted_error(df_target_mean_encoding):
	"""when fit is not called before transform."""
	with pytest.raises(NotFittedError):
	    transformer = TargetMeanEncoding(target='survived', 
			                         	variables_type=True)
	    transformer.transform(df_target_mean_encoding)
