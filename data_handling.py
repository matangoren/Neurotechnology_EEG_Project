import numpy as np
import pandas as pd
import seaborn
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import PowerTransformer

data_pain = pd.read_csv("data/pain_db.csv")

label_mapping = {"level_zero": 'Base', "level_one": 'DELETE', "level_two": 'Minor', "level_three": 'DELETE',
                 "level_four": 'Severe'}

data_pain = data_pain.replace({"Label": label_mapping})
data_pain = data_pain[data_pain.Label != 'DELETE']

# means = df.groupby(['Subject-ID', 'Label']).mean()

features = ['Subject_ID', 'Label', 'CH22_Sim_corr', 'CH22_S_sd', 'CH23_A_PEAK', 'CH23_Sim_corr', 'CH23_Sim_MutInfo',
            'CH24_Sim_corr', 'CH25_meanRR', 'CH25_rmssd', 'CH26_A_PEAK', 'CH26_Sim_corr']

pain_means = data_pain.groupby(['Subject_ID', 'Label'], as_index=False).mean()

pain_means = pd.DataFrame(pain_means, columns=features)

live_data = data_pain.groupby(['Subject_ID'], as_index=False).first()

data_pain.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN
data_pain.dropna(inplace=True)


# pca = PCA(n_components=2, whiten=True).fit([19, 151])
clusters = pd.DataFrame(columns=['Subject_ID', 'Label', 'Cluster'])

for subject, subject_data in data_pain.groupby('Subject_ID'):
    for label, subject_label_data in subject_data.groupby('Label'):
        data = subject_label_data.iloc[1:, 2:].to_numpy()
        cluster = pd.DataFrame(np.array([subject, label, [PCA(n_components=2, whiten=True).fit_transform(data)]]),
                               columns=['Subject_ID', 'Label', 'Cluster'])
        clusters = clusters.append(cluster, ignore_index=True)
print(clusters.shape)


# X = pd.DataFrame(data_pain, columns=features)
# Y = pd.DataFrame(data_pain, columns=['Label'])

