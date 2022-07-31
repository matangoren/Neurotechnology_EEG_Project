import numpy as np
import pandas as pd
import seaborn
from sklearn.preprocessing import PowerTransformer

data_pain = pd.read_csv("data/pain_db.csv")

label_mapping = {"level_zero": 'No Pain', "level_one": 'DELETE', "level_two": 'Minor', "level_three": 'DELETE',
                 "level_four": 'Severe'}

data_pain = data_pain.replace({"Label": label_mapping})
data_pain = data_pain[data_pain.Label != 'DELETE']

# means = df.groupby(['Subject-ID', 'Label']).mean()

means = data_pain.groupby(['Subject_ID', 'Label'], as_index=False).mean()



# means = (df.groupby(['Subject-ID']).mean().groupby(['Label']).mean())


features = ['CH22_Sim_corr', 'CH22_S_sd', 'CH23_A_PEAK', 'CH23_Sim_corr', 'CH23_Sim_MutInfo',
           'CH24_Sim_corr', 'CH25_meanRR', 'CH25_rmssd', 'CH26_A_PEAK', 'CH26_Sim_corr']



# Y = Y.replace({"Label": label_mapping})
#
# for subject, df_subject in df.groupby('Subject-ID'):
#     for label
#
#
# X = pd.DataFrame(df, columns=features)
# Y = pd.DataFrame(df, columns=['Label'])

