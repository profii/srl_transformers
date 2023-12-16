import pandas as pd
import numpy as np
import torch
import re
import os
import wandb
from tqdm import tqdm

from torch.utils.data import Dataset
from typing import List, Dict, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer
# from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration

from datasets import load_dataset, load_metric
from more_itertools import locate


BIO = ['person', 'organization', 'location', 'miscellaneous']
PATTERN = ["###", "Instruction:", "Find", "all", 'person', ',', 'organization', ',', 'location', 'and', 'miscellaneous', ".\n\n###", "Input:"]
END_PATTERN = ["\n\n###", "Response:"]

class PairsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        assert idx <= len(self.x['input_ids']), (idx, len(self.x['input_ids']))
        item = {key: val[idx] for key, val in self.x.items()}

        item['labels'] = self.y['input_ids'][idx]
        if IS_ENCODER_DECODER: item['decoder_attention_mask'] = self.y['attention_mask'][idx]

        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n


class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True
        )

        if IS_ENCODER_DECODER:
            ybatch = self.tokenizer.pad(
                {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
                padding=True
            )
        else:
            ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels']},
            padding=True
            )

        batch['labels'] = ybatch['input_ids']

        if IS_ENCODER_DECODER: batch['decoder_attention_mask'] = ybatch['attention_mask']


        return {k: torch.tensor(v) for k, v in batch.items()}


def mapping_data(data):
    sep = '_nan'
    token_list = []
    label_list = []

    for sent in data['tokens']:
        for t in sent:
            token_list.append(t.lower())
        token_list.append(sep)

    for sent in data['ner_tags']:
        for t in sent:
            label_list.append(t)
        label_list.append(sep)

    return pd.DataFrame(data={'data':token_list, 'label':label_list})


def separate_text(df):
    # Separating data into sentences with empty lines (NaN)

    a_pattern = ['|', BIO[0],']']
    b_pattern = ['|', BIO[1],']']
    c_pattern = ['|', BIO[2],']']
    d_pattern = ['|', BIO[3],']']
    sep = ','

    inp = [] # for inp
    out = []
    sentence = []
    prev_tag = 0
    temp_a = False
    temp_b = False
    temp_c = False
    temp_d = False
    a = []
    b = []
    c = []
    d = []

    for word, tag in df.values:
        if word == '_nan':
            inp.append(PATTERN + sentence + END_PATTERN)
            if len(a) != 0 and a[-1] == sep: del a[-1]
            if len(b) != 0 and b[-1] == sep: del b[-1]
            if len(c) != 0 and c[-1] == sep: del c[-1]
            if len(d) != 0 and d[-1] == sep: del d[-1]
            out.append(['['] + a + a_pattern + ['['] + b + b_pattern + ['['] + c + c_pattern + ['['] + d + d_pattern)
            sentence = []
            a = []
            b = []
            c = []
            d = []
            temp_a = False
            temp_b = False
            temp_c = False
            temp_d = False
            prev_tag = 0
        else:
            # tag = tag.lower()
            word = re.sub(r"[\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»\№]", ",", word)
            word = re.sub(r"[\,]+", ",", word)
            word = re.sub(r"[\.]+", ".", word)

            # If prev tag was the last one in a tag set
            if prev_tag != tag:
                if temp_a:
                    a.append(sep)
                    temp_a = False
                if temp_b:
                    b.append(sep)
                    temp_b = False
                if temp_c:
                    c.append(sep)
                    temp_c = False
                if temp_d:
                    d.append(sep)
                    temp_d = False

            if tag != 0:
                if tag == 1 or tag == 2:
                    a.append(word)
                    temp_a = True
                if tag == 3 or tag == 4:
                    b.append(word)
                    temp_b = True
                if tag == 5 or tag == 6:
                    c.append(word)
                    temp_c = True
                if tag == 7 or tag == 8:
                    d.append(word)
                    temp_d = True

            prev_tag = tag
            sentence.append(word)

    return inp, out


def separate_text_end(df):
    # Separating data into sentences with empty lines (NaN)

    # PATTERN = ["###", "Instruction:", "Find", "all", "aspects", ",", "objects", "and", "predicates", ".\n\n###", "inp:"]
    PATTERN = ["###", "Instruction:", "Find", "all", 'person', ',', 'organization', ',', 'location', 'and', 'miscellaneous', ".\n\n###", "Input:"]
    END_PATTERN = ["\n\n###", "Response:"]
    inp = []
    sentence = []

    for word in df['data']:
        if word == '_nan':
            inp.append(PATTERN + sentence + END_PATTERN)
            sentence = []
        else:
            word = re.sub(r"[\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»\№]", ",", word)
            word = re.sub(r"[,]+", ",", word)
            word = re.sub(r"[.]+", ".", word)

            sentence.append(word)

    return inp


def evaluate(dfo):
    # indexes_nan = []
    labels_list = []
    sents = separate_text_end(dfo)
    i = 0

    for sent in tqdm(sents):
        input_ids = tokenizer.encode(sent, return_tensors="pt", is_split_into_words=True)

        outs = model.generate(input_ids.to("cuda"), no_repeat_ngram_size=6,
                                max_new_tokens=2048,
                                num_return_sequences=1, early_stopping=True)

        decoded = tokenizer.decode(outs[0], skip_special_tokens=True)

        if i < 8:
            i += 1
            print(len(decoded))
            # print(len(sent))
            print(decoded)
            print('--------')
            # print(labels_list)

        labels_list.append(decoded)

    return labels_list


if __name__ == "__main__":
    datasets = load_dataset("conll2003")
    df = mapping_data(datasets["train"])
    df_dev = mapping_data(datasets["validation"])
    
    metric = load_metric("seqeval")
    label_list = datasets["train"].features["ner_tags"].feature.names

    print('Loaded data')
    print(df.shape, df_dev.shape)

    # Appling cleaning to df
    inp, out = separate_text(df)
    inp_dev, out_dev = separate_text(df_dev)

    print('\nExample:')
    print(' '.join(inp[-1]))
    print(' '.join(out[-1]))

    ### T5 # 6ep_8b
    MODEL_NAME = 't5-large'
    IS_ENCODER_DECODER = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to('cuda')
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to('cuda')

    # ### FLAN-T5 # 4ep_8b
    # MODEL_NAME = 'flan-t5'
    # MODEL_HF_NAME = 'google/flan-t5-large'
    # IS_ENCODER_DECODER = True
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
    # model = T5ForConditionalGeneration.from_pretrained(MODEL_HF_NAME).to('cuda')
    # tokenizer.eos_token = '</s>'

    # CodeT5
    # Salesforce/codet5-small

    MAX_LENGTH = 128

    train_dataset = PairsDataset(tokenizer(inp, padding='max_length', max_length=MAX_LENGTH, is_split_into_words=True),
                                tokenizer(out, padding='max_length', max_length=MAX_LENGTH, is_split_into_words=True))
    dev_dataset = PairsDataset(tokenizer(inp_dev, padding='max_length', max_length=MAX_LENGTH, is_split_into_words=True),
                            tokenizer(out_dev, padding='max_length', max_length=MAX_LENGTH, is_split_into_words=True))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    os.environ["WANDB_PROJECT"] = '<nlp-project>'

    N_EPOCHS = 8
    BATCH_SIZE = 8
    run_name = f"CoNLL_{MODEL_NAME}_{N_EPOCHS}ep_{BATCH_SIZE}b"
    saved_name = '_'.join([MODEL_NAME, str(N_EPOCHS)+'ep', str(BATCH_SIZE)+'b'])

    args = TrainingArguments(output_dir="experiments/"+saved_name,
                            num_train_epochs=N_EPOCHS,
                            per_device_train_batch_size=BATCH_SIZE,
                            per_device_eval_batch_size=BATCH_SIZE,
                            #  save_steps=10000000,
                            logging_steps=100,
                            report_to="wandb",  # enable logging to W&B
                            run_name=run_name,  # name of the W&B run (optional)
                            load_best_model_at_end = True,
                            evaluation_strategy = 'epoch',
                            save_strategy = "epoch",
                            save_total_limit = 3,
                            #  optim='adamw_torch',
                            #  learning_rate=5e-5,
                            weight_decay=0.01,
                            seed=42
                            )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        # compute_metrics=compute_metrics
    )

    print('Start to train')

    trainer.train()

    wandb.finish()

    df_testo = mapping_data(datasets["test"])
    df_test = df_testo[df_testo['data'] != '_nan']

    df_devo = df_dev.copy(deep=True)
    df_dev = df_dev[df_dev['data'] != '_nan']

    print('Dev:')
    print(df_dev.shape, df_devo.shape)
    
    model_name = 'dev'
    labels_list = evaluate(df_devo)
    
    with torch.no_grad():
        labels = []
        i = 0
        sent = []
        labeled_sent = [] # for out
        tags = []
        BIO = [1, 2, 3, 4, 5, 6, 7, 8]

        for d in tqdm(df_devo.data):
            d = re.sub(r"[\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»\№]", ",", d)
            d = re.sub(r"[,]+", ",", d)
            d = re.sub(r"[.]+", ".", d)
            
            # co += 1
            if d == '_nan':
                labels = labels_list[i].split('] [')
                if len(labels) < 4:
                  n = len(labels)
                else:
                  n = 4

                for l in range(n):
                    word_list = labels[l].split('|')[0].replace('[', '').strip().split(',')

                    for j in word_list:
                        j = j.strip()
                        if ' ' in j:
                            for beg, elem in enumerate(j.split()):
                                if beg == 0:
                                    if elem in sent: tags[sent.index(elem)] = BIO[2*l]
                                else:
                                    if elem in sent: tags[sent.index(elem)] = BIO[2*l+1]
                        elif j in sent:
                            indices = locate(sent, lambda x: x == j)

                            for inde in indices:
                                tags[inde] = BIO[2*l]

                i += 1
                labeled_sent.extend(tags)
                sent = []
                tags = []
            else:
                sent.append(d.lower())
                tags.append(0)    
    
        df_dev['labels'] = labeled_sent # dev
        print(df_dev.head(30))
    
        df_dev.to_csv('experiments/'+saved_name+'/'+saved_name+'_'+model_name+'.tsv',
                        header=None, index=False, quoting=3, sep='\t', encoding='utf-8')

        pred = [label_list[i] for i in df_dev['labels']]
        ref = [label_list[i] for i in df_dev['label']]
        
        res = metric.compute(predictions=[pred], references=[ref])
        print(res)

        with open('experiments/'+saved_name+'/'+saved_name+'_'+model_name+'_metrics.txt', 'w') as out:
            for key in res:
                out.write(key)
                out.write('\n')
                out.write(res[key])
                out.write('\n')
            print('A miracle happened ^-^/***')




    print('Test:')
    print(df_test.shape, df_testo.shape)

    model_name = 'test'
    labels_list = evaluate(df_testo)

    with torch.no_grad():
        labels = []
        i = 0
        sent = []
        labeled_sent = [] # for out
        tags = []
        BIO = [1, 2, 3, 4, 5, 6, 7, 8]

        for d in tqdm(df_testo.data):
            d = re.sub(r"[\"\—\#\$\%\&\'\(\)\*\+\,\–\-\/\:\;\<\=\>\?\@\[\\\]\^\?\!\_\`\{\|\}\~\«\»\№]", ",", d)
            d = re.sub(r"[,]+", ",", d)
            d = re.sub(r"[.]+", ".", d)
            
            # co += 1
            if d == '_nan':
                labels = labels_list[i].split('] [')
                
                if len(labels) < 4:
                  n = len(labels)
                else:
                  n = 4

                for l in range(n):
                    word_list = labels[l].split('|')[0].replace('[', '').strip().split(',')

                    for j in word_list:
                        j = j.strip()
                        if ' ' in j:
                            for beg, elem in enumerate(j.split()):
                                if beg == 0:
                                    if elem in sent: tags[sent.index(elem)] = BIO[2*l]
                                else:
                                    if elem in sent: tags[sent.index(elem)] = BIO[2*l+1]
                        elif j in sent:
                            indices = locate(sent, lambda x: x == j)

                            for inde in indices:
                                tags[inde] = BIO[2*l]

                i += 1
                labeled_sent.extend(tags)
                sent = []
                tags = []
            else:
                sent.append(d.lower())
                tags.append(0)    
    
        df_test['labels'] = labeled_sent # test
        print(df_test.head(30))

        df_test.to_csv('experiments/'+saved_name+'/'+saved_name+'_'+model_name+'.tsv',
                        header=None, index=False, quoting=3, sep='\t', encoding='utf-8')

        pred = [label_list[i] for i in df_test['labels']]
        ref = [label_list[i] for i in df_test['label']]

        res = metric.compute(predictions=[pred], references=[ref])
        print(res)

        with open('experiments/'+saved_name+'/'+saved_name+'_'+model_name+'_metrics.txt', 'w') as out:
            for key in res:
                out.write(key)
                out.write('\n')
                out.write(res[key])
                out.write('\n')
            print('A miracle happened ^-^/***')



