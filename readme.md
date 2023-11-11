# Peeking into the Black Box: Interpreting NN Classification Using Heatmaps

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig1.svg?raw=true)
## Motivation

Various models can be trained to classify marine or terrestrial natural products using data aggregated in the COCONUT database. These models aren't very useful, but they beg an interesting question: *how are they telling the natural products apart, anyway?*

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig7.svg?raw=true)

*The discovery and development of marine compounds with pharmaceutical potential.* Munro et al. (1999)

The National Cancer Institute (NCI) has shown that marine natural products extracts have much higher hit-rates (i.e. bioactivity) than terrestrial extracts. Is it possible to figure out, structurally, what makes those extracts so different?

This project was an attempt to peek into the black box, and extract human-interpretable insights.
## Method

The COCONUT database collates data from many other databases, and many entries are labelled by origin. Terrestrial natural products comprise the vast majority of entries, so the model was trained on a randomly selected subset of 50% terrestrial and 50% marine NPs. The dataset is also dominated by molecules labelled as plant-origin, however I did not attempt to balance these because I was concerned about molecules produced by microbial symbionts.

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/dataset.svg?raw=true)

In order to infer which parts of a given molecule contribute to it's label, one can systematically omit different parts and observe the effects on the classification output. This can be used to construct a 'heatmap' highlighting which parts of a molecule contribute the most to it's classification.

My intuition was that this would work better for models trained on features which have a close relationship with the molecular structure. I wanted to avoid the common chemical fingerprints because I needed the systematically modified molecules to still result in similar classifications to the originals. Therefore, I used a graph convolutional neural network (GCNN). Everything described above was performed using the [DeepChem](https://deepchem.io/) library (*Deep Learning for the Life Sciences.* Ramsundar et al. 2019)

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig2.svg?raw=true)

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig3.svg?raw=true)

## Results

The large discrepancy in model performance between the training and test data suggests that the model is overfitting. One way to improve this would be by optimizing hyperparameters.

That aside, things seem to have worked fairly well. Inspecting a random sample of the dataset shows things that match intuition:

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig4.svg?raw=true)

Were I to guess and classify these molecules myself, I probably would have done it based on those atoms. However, closer inspection of the data reveals lots of completely mangled structures:

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig5.svg?raw=true)

Given the common adage, 'Garbage In - Garbage Out', this bodes poorly for any insights one could hope to glean from this exercise. Pretty easy fix though: just find better data and re-do everything.

Finally, the model shows some really strange and counter-intuitive heatmaps:

![](https://github.com/mhagar/marine-heatmaps/blob/main/chemfigs/fig6.svg?raw=true)

It's possible this is a result of overfitting. However, these 'counter-intuitive' heatmaps are perhaps the most interesting part of the exercise. Could it be that there are unrecognized patterns that distinguish marine NPs from terrestrial ones? Could it be that, this whole time, Nature expressed different 'architectural movements' on land and in the sea?

Hopefully, I'll get to pick up on this project again someday.



