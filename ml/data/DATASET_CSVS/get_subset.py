import pandas as pd

df = pd.read_csv('deepship_5k_seg_3s.csv')
grouped = df.groupby(['labels', 'folds'])
subset_df = grouped.sample(n=15, random_state=42)
subset_df.to_csv('deepship_5k_seg_3s_tiny.csv')
