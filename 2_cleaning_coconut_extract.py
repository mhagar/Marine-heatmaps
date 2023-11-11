import pandas as pd
import numpy as np
import ast
import re

import rdkit.Chem as rdkc
from rdkit import rdBase

import multiprocessing as mp
from tqdm import tqdm

# Suppresses rdkit errors when SMILE code is broken
rdBase.DisableLog('rdApp.error')

# Preparing the DataFrame
df = pd.read_csv('coconut_extract.csv', index_col=0)

# Progress bar for keeping track
pbar = tqdm(total=len(df))

# Retain these classes in the ORIGIN column
keep_these = ['plants', 'bacteria', 'fungi', 'animals', 'marine']
key = {'plants': 0,
       'bacteria': 1,
       'fungi': 2,
       'animals': 3,
       'marine': 4}

strip_pattern = re.compile(r'\[|\]| ')


def parse(df):
    for i, k in df.iterrows():
        row = df.loc[i]

        # PROCESSING 'ORIGIN' COLUMN
        # the data is stored as string: "['plant', 'marine']".
        j = ast.literal_eval(row['origin'])  # evaluate str into python list
        j = sorted([key[taxon] for taxon in j if taxon in keep_these])
        j = re.sub(strip_pattern, '', str(j))  # replaces '[0, 4, ..]' with '0,4,..'
        df.loc[i, 'origin'] = j  # replace w string '4,3,1' referring to key

        # PROCESSING 'FOUND_IN_DATABASES' COLUMN
        # the data is stored just like ORIGIN column.
        j = ast.literal_eval(row['found_in_databases'])
        j = sorted(j)
        j = re.sub(strip_pattern, '', str(j))  # replaces '[0, 4, ..]' with '0,4,..'
        df.loc[i, 'found_in_databases'] = j

        pbar.update(1)
    return df


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


parsed_df = parallelize_dataframe(df, parse)

# Now to go through and delete invalid SMILES strings
# I know I should've done this during the multi-core parse above, but i couldnt
# figure out how to do get multiple processes to manipulate the same list
# and it's not worth my time right now to figure out how to do it

invalid_rows = []
pbar = tqdm(total=len(parsed_df))

for i, k in parsed_df.iterrows():
    row = parsed_df.loc[i]
    m = rdkc.MolFromSmiles(row['smiles'])
    if m is None:
        invalid_rows.append(i)
        pbar.update(1)
        continue
    pbar.update(1)


# Now invalid_rows is a list of indices in parsed_df with invalid SMILES
# Popping them:
parsed_df.drop(invalid_rows, axis=0, inplace=True)

with open('processed_coc_extct_invalid_smiles.txt', 'w') as f:
    for item in invalid_rows:
        f.write('%s\n' % item)

parsed_df.to_csv('processed_coconut_extract.csv')
