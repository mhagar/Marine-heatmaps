import deepchem as dc
import pandas as pd
import numpy as np
np.random.seed(69)

df = pd.read_csv('processed_coconut_extract.csv', index_col=0)
df.dropna(subset=['origin'], inplace=True)  # Drops items with no label

marine_df = df[df['origin'].str.contains('4')]
marine_df.insert(loc=len(marine_df.columns),
                 column="marine",
                 value=[1]*len(marine_df))

terra_df = df[~df['origin'].str.contains('4')]

# Random list of rows to keep:
keep_these = np.random.choice(terra_df.index, len(marine_df), replace=False)
trunc_terra_df = terra_df.loc[keep_these]
trunc_terra_df.insert(loc=len(trunc_terra_df.columns),
                      column="marine",
                      value=[0]*len(marine_df))

balanced_df = pd.concat([marine_df, trunc_terra_df])
balanced_df.to_pickle("balanced_df.pkl")

import pdb; pdb.set_trace()
weights = [1]*len(balanced_df)
ids = balanced_df['coconut_id']

featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
# If you flip this to "True", the data_loader bugs out.

loader = dc.data.InMemoryLoader(tasks=["marine"],
                                featurizer=featurizer)

dataset = loader.create_dataset(inputs=zip(balanced_df['smiles'],
                                           balanced_df['marine'],
                                           weights,
                                           ids),
                                data_dir='Marine_DataDisk',
                                shard_size=1000)
