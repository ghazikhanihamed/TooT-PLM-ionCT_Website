# TooT-PLM-ionCT

## Introduction
**TooT-PLM-ionCT** is a pioneering framework designed for the precise classification of ion channels (ICs) and ion transporters (ITs), crucial for distinguishing between these two essential types of membrane proteins. Leveraging the power of advanced Protein Language Models (PLMs) such as **ESM-1b** and **ESM-2**, combined with sophisticated machine learning classifiers like logistic regression and Convolutional Neural Networks (CNN), TooT-PLM-ionCT sets a new standard in the bioinformatics field for understanding membrane protein functions without the extensive need for laboratory experiments.

## Dataset
The dataset pivotal to the TooT-PLM-ionCT framework is accessible at [Hugging Face Datasets](https://huggingface.co/datasets/ghazikhanihamed/TooT-PLM-ionCT_DB). This dataset is integral to the training and evaluation of the models within the TooT-PLM-ionCT framework.

## Framework Overview
TooT-PLM-ionCT employs two distinct approaches based on the task:

- **For IC-MP and IT-MP classification**: Utilizes the **ESM-1b** model combined with a logistic regression classifier, providing a robust method for categorizing ion channels and other membrane proteins, as well as ion transporters and membrane proteins.
  
- **For IC-IT differentiation**: Integrates the **ESM-2** model with a Convolutional Neural Network (CNN) classifier, specifically tailored to distinguish between ion channels and ion transporters effectively.

This methodical application of PLMs and machine learning techniques aims to facilitate the rapid understanding of these proteins' roles, significantly propelling forward bioinformatics research.
