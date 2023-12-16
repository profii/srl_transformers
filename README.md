# Semantic Role Labeling with Seq2seq Transformers and BiLSTM Architecture

The main idea of this project is to implement sequence-to-sequence transformers for Semantic Role Labeling classification task and try BiLSTM Architecture.

We use two datasets:

-- Codalab challenge (https://codalab.lisn.upsaclay.fr/competitions/531);

-- CoNLL-2003 (Sang and De Meulder, 2003).

Seq2seq encoder-decoder models: T5 and FLAN-T5.

Achitecture: BiLSTM.

F1-score for Codalab challenge on Dev: **0.742** (FLAN-T5), 0.740 (T5).

F1-score for Codalab challenge on Test: **0.863** (FLAN-T5), 0.862 (T5).

Example of T5 model's input and output on CoNLL dataset:

| Input  | Output |
| ------------- | ------------- |
|\#\#\# Instruction: Find all person , organization , location and miscellaneous .|
|\#\#\# Input: After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 . |
| \#\#\# Response: |  [ andy, caddick \| person ] [ somerset, leicestershire \| organization ] [ grace, road, england \| location ] [ \| miscellaneous ] |
