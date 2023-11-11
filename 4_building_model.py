import deepchem as dc
import numpy as np
import pprint as pp


def yes_or_no(question):
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if reply[:1] == 'y':
        return True
    elif reply[:1] == 'n':
        return False
    else:
        return yes_or_no("Invalid answer, try again")


# Load Dataset
total_data = dc.data.DiskDataset(data_dir='Marine_DataDisk')
splitter = dc.splits.RandomSplitter()
tr_data, val_data, test_data = splitter.train_valid_test_split(
                                            dataset=total_data,
                                            frac_train=0.8,
                                            frac_valid=0.1,
                                            frac_test=0.1,
                                            seed=69)
# Metrics:
eval_these = [dc.metrics.recall_score,
              dc.metrics.precision_score]

metrics = [dc.metrics.Metric(x,
                             np.mean,
                             mode="classification",
                             classification_handling_mode="threshold",
                             threshold_value=0.5) for x in eval_these]

metrics.append(dc.metrics.Metric(dc.metrics.roc_auc_score,
                                 np.mean,
                                 mode="classification"))

# Fit/Load Model:
model = dc.models.GraphConvModel(1, mode='classification', model_dir='model')
if yes_or_no("Train model from scratch?"):
    model.fit(tr_data, nb_epoch=50)
else:
    model.restore(model_dir='model')

# Initial Evaluation:
train_scores, val_scores, test_scores = tuple(model.evaluate(x, metrics)
                                              for x in [tr_data,
                                                        val_data,
                                                        test_data])
if yes_or_no("View evaluation?"):
    print("Evaluating training data:")
    pp.pprint(train_scores)
    print("Evaluating validation data:")
    pp.pprint(val_scores)
    print("Evaluating test data:")
    pp.pprint(test_scores)

# TODO: implement hyperparameter tuning
