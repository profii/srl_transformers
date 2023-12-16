# Semantic Role Labeling with Seq2seq Transformers and BiLSTM Architecture
This repository contains the code and datasets used in our Semantic Role Labeling (SRL) project. The main idea of this project is to implement sequence-to-sequence transformers for Semantic Role Labeling classification task and try BiLSTM Architecture.

We use two datasets:

-- Codalab challenge (https://codalab.lisn.upsaclay.fr/competitions/531);

-- CoNLL-2003 (Sang and De Meulder, 2003).

Seq2seq encoder-decoder models: T5 and FLAN-T5.

Achitecture: BiLSTM.


## Repository Structure

### Dataset Folder
- `dataset/`: This folder contains the data used in our project, sourced from the Codalab Challenge. It includes annotated sentences with labels in BIO format, covering entities such as "Object", "Aspect", and "Predicate".

### Python Scripts
- `conll.py`: This script is responsible for handling the CoNLL 2003 dataset. The CoNLL 2003 dataset is a benchmark dataset in Named Entity Recognition and is used in our project to supplement our primary dataset from the Codalab Challenge.

### Jupyter Notebooks
- `Seq2seq_t5.ipynb`: This Jupyter Notebook contains the code for T5. It demonstrates the implementation details and the training process of the model on the SRL task.
- `srl_bert.ipynb`: This notebook implements the BERT model for SRL, including preprocessing, model training, and evaluation.
- `srl_bilstm.ipynb`: This file contains the BiLSTM model implementation for SRL, detailing the architecture, training, and performance evaluation.

## Model Scores
Here we present the best performance scores of our models on the SRL task:
F1-score for Codalab challenge on Dev: **0.742** (FLAN-T5), 0.740 (T5).

F1-score for Codalab challenge on Test: **0.863** (FLAN-T5), 0.862 (T5).

## Input/Output Examples
Below are some examples demonstrating the input to our models and their corresponding outputs:


Example of T5 model's input and output on CoNLL dataset:

| Input  | Output |
| ------------- | ------------- |
|\#\#\# Instruction: Find all person , organization , location and miscellaneous .|
|\#\#\# Input: After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 . |
| \#\#\# Response: |  [ andy, caddick \| person ] [ somerset, leicestershire \| organization ] [ grace, road, england \| location ] [ \| miscellaneous ] |


## License

This project is open-sourced under the [MIT License](LICENSE).




