import deepchem as dc
import pandas as pd
import numpy as np
import rdkit.Chem as rdkc
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
np.random.seed(69)
rdkc.Draw.rdMolDraw2D.MolDrawOptions().useBWAtomPalette()

# ### Load Dataset
total_data = dc.data.DiskDataset(data_dir='Marine_DataDisk')
splitter = dc.splits.RandomSplitter()
tr_data, val_data, test_data = splitter.train_valid_test_split(
                                            dataset=total_data,
                                            frac_train=0.8,
                                            frac_valid=0.1,
                                            frac_test=0.1,
                                            seed=69)

# ### Load Dataframe
df = pd.read_pickle('balanced_df.pkl')

# ### Load CNN
model = dc.models.GraphConvModel(1, mode='classification', model_dir='model')
model.restore()

# ### Visualizer Function
legend = {0: 'Terra', 1: 'Marine'}
full_featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
frag_featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True)
#   setting "per_atom_fragmentation=True" returns a list of ConvMols,
#   sequentially omitting each atom


def visualize(idx, dataset):
    c_id = dataset.ids[idx]  # Fetch coconut ID
    row = df.loc[df["coconut_id"] == c_id].squeeze()  # Fetch df row as series
    smiles = row['smiles']
    mol = rdkc.MolFromSmiles(smiles)

    frag_set = frag_featurizer.featurize(smiles)
    frag_dset = dc.data.NumpyDataset(np.array(frag_set[0]))
    main_mol = full_featurizer.featurize(smiles)
    main_mol_dset = dc.data.NumpyDataset(np.array(main_mol))

    frag_preds = model.predict(frag_dset)
    frag_preds = np.squeeze(frag_preds)[:, 1]
    #   list of probabilities each fragment is predicted to be class 1
    main_pred = model.predict(main_mol_dset)  # probability of whole molecule
    main_pred = np.squeeze(main_pred)[1]

    contribs = frag_preds - main_pred
    atom_idxs = rdkc.rdmolfiles.CanonicalRankAtoms(mol)
    weights = dict(zip(atom_idxs, contribs))
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, alpha=0.1,
                                                     colorMap='BrBG_r')

    caption = f"{c_id} \n"\
              f"Prediction: {main_pred:.1%} Marine \n"\
              f"True Origin: {legend[row['marine']]} "
    plt.text(0.01, 0.02, caption, fontsize=14,
             bbox=dict(facecolor='red', alpha=0.1))
    fig.savefig(f'figures/{c_id}.png', bbox_inches='tight')
    plt.close()


def visualize_mol(smiles, c_id, origin, dir):
    mol = rdkc.MolFromSmiles(smiles)

    frag_set = frag_featurizer.featurize(smiles)
    frag_dset = dc.data.NumpyDataset(np.array(frag_set[0]))
    main_mol = full_featurizer.featurize(smiles)
    main_mol_dset = dc.data.NumpyDataset(np.array(main_mol))

    frag_preds = model.predict(frag_dset)
    frag_preds = np.squeeze(frag_preds)[:, 1]
    #   list of probabilities each fragment is predicted to be class 1
    main_pred = model.predict(main_mol_dset)  # probability of whole molecule
    main_pred = np.squeeze(main_pred)[1]

    contribs = frag_preds - main_pred
    atom_idxs = rdkc.rdmolfiles.CanonicalRankAtoms(mol)
    weights = dict(zip(atom_idxs, contribs))
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, alpha=0.1,
                                                     colorMap='BrBG_r')

    caption = f"{c_id} \n"\
              f"Prediction: {main_pred:.1%} Marine \n"\
              f"True Origin: {origin} "
    plt.text(0.01, 0.02, caption, fontsize=14,
             bbox=dict(facecolor='red', alpha=0.1))
    fig.savefig(f'{dir}/{c_id}.png', bbox_inches='tight')
    plt.close()


# ### Generating a number of random figures
# if __name__ == "__main__":
#     random_idxs = np.random.randint(0, len(total_data), size=500)
#     for idx in random_idxs:
#         visualize(idx, total_data)
