import bson
import pandas as pd
from progress.bar import FillingSquaresBar

"""
Mar 31 2021
Extracts the following features from dataset:
- coconut_id
- smiles
- chemicalSubClass
- found_in_databases <list>
- textTaxa <list> (marine, etc)
- name
"""

# sourceNP.bson contains the unconsolidated sum of all the cococnut datasets
# you wanna work with uniqueNP.bson, thats been consolidated & has metadata
data = bson.decode_file_iter(open('COCONUTlatest/COCONUT2020-10/'
                                  'uniqueNaturalProduct.bson', 'rb'))

bar = FillingSquaresBar('Counting..', max=400000)
extracted_data = []
for d in data:
    row = {}
    row['coconut_id'] = d['coconut_id']
    row['name'] = d['name']
    row['smiles'] = d['smiles']
    row['chemicalSubClass'] = d['chemicalSubClass']
    row['found_in_databases'] = d['found_in_databases']
    row['origin'] = d['textTaxa']

    extracted_data.append(row)
    bar.next()

df = pd.DataFrame(extracted_data)
df.to_csv('coconut_extract.csv')

import pdb; pdb.set_trace()
