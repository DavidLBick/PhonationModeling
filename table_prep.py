import pandas as pd 
import numpy as np

# read results_full_class_info-theory.txt as csv 
df = pd.read_csv('results_full_class_info-theory.txt', sep=',', header=None)

# read results_full_symptomatic_info-theory.txt as csv
df2 = pd.read_csv('results_full_symptomatic_info-theory.txt', sep=',', header=None)

# set column names as step_size, time_series, recording, feature, p_value
df.columns = ['step_size', 'time_series', 'recording', 'feature', 'p_value']
df2.columns = ['step_size', 'time_series', 'recording', 'feature', 'p_value']

# group by step_size and return the top 2 step-sizes by max count 
df.groupby('step_size').count().sort_values('time_series', ascending=False).head(2)
df2.groupby('step_size').count().sort_values('time_series', ascending=False).head(2)

breakpoint()