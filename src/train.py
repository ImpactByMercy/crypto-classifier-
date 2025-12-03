import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import joblib

def prepare_data(df_features,df_lables):
  #loading data
  df1= df_features
  df2= df_lables["label"]
  #merging features and labels
  if len(df_features) != len (df_lables):
    df_merged =pd.concat([df_features,df_labels],axis=1,join="inner")
  else:
    df_merged =pd.concat([df_features,df_labels],axis=1)
  #dropping na values
  df_clean= df_merged.dropna()
  columns =["dist_sma_7", "dist_sma_20", "dist_sma_200", "bb_pct_b", "rsi", "volume_ratio","close_open_pct","high_low_pct","macd_diff","label"]
  df_processed = df_clean[columns]
  x = df_processed.drop("label",axis=1)
  y = df_processed["label"]

  print(f"successfull",x.info(),y.info())

  return x,y
def train_model(X,y):
  pass



