""" 
  This module synthesizes some of the functions using on the major algorithms of this 
  project, as a way to keep the main code as clean as possible for analysis. And here
  the APIs for each one of this functions is presented.
"""

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from datetime import datetime
from sklearn import preprocessing
from scipy.optimize import dual_annealing
from sklearn.metrics import confusion_matrix


def replaceFields(dataset=None):
  """
    This function is responsible to replace some data features
    with ones that are more suitable considering the analysis 
    porpuse, such as: 
    
    * Birth Year => Age in years (int)
    * Dt Customer => Persistance in months (int)
    
    :param pandas.DataFrame dataset: The dataset table as dataframe

    :return: The dataset with some fields preprocessed
    :rtype: pandas.DataFrame
  """
  features = dataset.keys()
  for feature in features:
    if feature == "Year_Birth":
      dataset[feature] = datetime.now().year - dataset[feature]
      dataset = dataset.rename(columns={feature: 'Age'})
    if feature == "Dt_Customer":
      in_dates = [datetime.fromisoformat(i) for i in dataset[feature].to_list()]
      dataset[feature] = [(datetime.now() - date).days/30 for date in in_dates]
      dataset = dataset.rename(columns={feature: 'Persistence'})
  return dataset

def encodeDataSet(dataset=None):
  """
    This function endodes the dataset into only numeric fields 
    for machine learning purpose. Returning the numerical dataset,
    with the encoders responsible for the transformation.

    :param pandas.DataFrame dataset: The dataset table as dataframe

    :return: The dataset with all categorical fields numerrically encoded together with a dictionary of each field encoder
    :rtype: pandas.DataFrame, dict
  """
  encoders = dict()
  features = dataset.keys()
  for feature in features:
    sample_value = dataset[feature][0]
    if type(sample_value) == str:
      le = preprocessing.LabelEncoder()
      le.fit(dataset[feature].to_list())
      dataset[feature] = le.transform(dataset[feature])
      encoders[feature] = le
  return dataset, encoders

def dropNonInformative(dataset=None):
  """
    This function search for inconsistency and stationarity inside
    the dataset, and then remove those informations from the dataset.
    It returns a dataset without features that does not provide any 
    interesting information... It also removes some 
    
    .. admonition:: Notice

      Notice that the dataset description has only 25 fields, but
      the table has 29 features... Probably some of those does not 
      have any information at all.
    
    :param pandas.DataFrame dataset: The dataset table as dataframe

    :return: The dataset with all non informative data dropped
    :rtype: pandas.DataFrame
  """
  del_features = list()
  features = dataset.keys()
  for feature in features:
      max_val = dataset[feature].max()
      min_val = dataset[feature].min()
      if max_val == min_val:
          del_features.append(feature)
  dataset = dataset.loc[:, ~dataset.columns.isin(del_features)]
  print("Features dropped:", del_features)
  return dataset

def balanceDataSet(phi, y):
  """
    This function will receive an output wise unbalaced regression
    model in the format 
    
    .. math::

      y(k) = f(\phi(k), \Theta )
    
    and will return a balanced dataset with randomized samples by
    considering the True label of the output as a reference. If 
    one wants to use the False label as reference, it is just 
    necessary to pass the ~ version of the output.
    
    The balanced database means a database with 50% of true targets,
    and 50% of false targets. This is interesting to remove bias of 
    upper/lower cut learning of the models.

    :param numpy.ndarray phi: The regressor matrix
    :param numpy.ndarray y: The targets vector

    :return: The new regression model => (phi, target)
    :rtype: tuple
  """
  t_indexes = np.where(y == 1)[0]
  f_indexes = np.where(y == 0)[0]
  t_target, f_target = y[t_indexes], y[f_indexes]
  t_phi, f_phi = phi[t_indexes,:], phi[f_indexes,:] 
  # Determine the model indexes
  n_samples, size = len(t_indexes), len(f_indexes)
  new_ind = np.random.randint(0, size, size=n_samples)
  f_target, f_phi = f_target[new_ind], f_phi[new_ind, :]
  new_phi = np.concatenate((t_phi, f_phi), axis=0)
  new_target = np.concatenate((t_target, f_target), axis=0)
  return new_phi, new_target


def xgbHyperGridSearch(bounds, data, iters=1000):
  """
    This function will run the annealing stochastic optimization algorithm
    based on the xgbCostFunction, to find the best set of hyper parameters
    for the XGBoost classifier, by minimizing the xgbCostFunction. 

    :param list bounds: The upper and lower bounds of each parameter.
    :param list data: The list of datasets that will be used in the xgbCostFunction.
    :param int iters: The number of maximun iterations on the annealing search.

    :return: The resulted parameters and the optimization summary, respectivelly.
    :rtype: tuple
  """
  res = dual_annealing(xgbCostFunction, maxiter=2000, bounds=list(bounds), args=data)
  pars = (int(res.x[0]), int(res.x[1]), res.x[2])
  return pars, res
    
def xgbCostFunction(p, yt, xt, yv, xv):
  """
    The cost function responsible to build a model with the provided set of 
    parameters, then estimate the model, and test its result in the testing 
    dataset. To then retrieve a performance indicator that will be the 
    reference for the optimization algorithm to minimize.

    :param list p: The set of hyper parameters candidates.
    :param numpy.ndarray yt: The train targets.
    :param numpy.ndarray xt: The train features.
    :param numpy.ndarray yv: The test targets.
    :param numpy.ndarray xv: The test features.

    :return: The sum of the false positive indicators from the confusion matrix.
    :rtype: float
  """
  # Define the parameters
  depth, weight = int(p[0]), int(p[1]) 
  gamma = p[2]
  # Build the model and estimate
  model = xgb.XGBClassifier(max_depth=depth, min_child_weight=weight, gamma=gamma)
  model.fit(xt, yt,
            eval_set=[(xt, yt), (xv, yv)],
            eval_metric='logloss', verbose=False)
  # Estimate the test output
  y_pred = model.predict(xv)
  # Compute the confusion matrix
  conf_mat = confusion_matrix(yv, y_pred, normalize='true')
  # Return the cost value to be minimized
  return conf_mat[0,1] + conf_mat[1,0]


def svmHyperGridSearch(bounds, data, iters=1000):
  """
    This function will run the annealing stochastic optimization algorithm
    based on the svmCostFunction, to find the best set of hyper parameters
    for the SVC classifier from sklearn, by minimizing the svmCostFunction. 

    :param list bounds: The upper and lower bounds of each parameter.
    :param list data: The list of datasets that will be used in the svmCostFunction.
    :param int iters: The number of maximun iterations on the annealing search.

    :return: The resulted parameters and the optimization summary, respectivelly.
    :rtype: tuple
  """
  res = dual_annealing(svmCostFunction, maxiter=2000, bounds=list(bounds), args=data)
  return res.x, res


def svmCostFunction(p, yt, xt, yv, xv):
  """
    The cost function responsible to build a model with the provided set of 
    parameters, then estimate the model, and test its result in the testing 
    dataset. To then retrieve a performance indicator that will be the 
    reference for the optimization algorithm to minimize.

    :param list p: The set of hyper parameters candidates.
    :param numpy.ndarray yt: The train targets.
    :param numpy.ndarray xt: The train features.
    :param numpy.ndarray yv: The test targets.
    :param numpy.ndarray xv: The test features.

    :return: The sum of the false positive indicators from the confusion matrix.
    :rtype: float
  """
  # Define the parameters
  C, gamma = p[0], p[1]
  # Build the model and estimate
  model = SVC(C=C, gamma=gamma, max_iter=200)
  model.fit(xt, yt)
  # Estimate the test output
  y_pred = model.predict(xv)
  # Compute the confusion matrix
  conf_mat = confusion_matrix(yv, y_pred, normalize='true')
  # Return the cost value to be minimized
  return conf_mat[0,1] + conf_mat[1,0]