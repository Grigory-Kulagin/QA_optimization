# -*- coding: utf-8 -*-
from typing import List, Optional, NoReturn

import pandas as pd
import torch
import time
import json
import torch.nn.functional as F
import numpy as np
from progressbar import progressbar as pb

"""Dataset preporation"""

def read_data(path, parts_num = 1, part = 1):
    """Divides dataframe into context, question and answer listst.

    Function can prepare as whole dataset as only its part.
    The dataset must contain columns: 'text', 'question', 'span_answer'.

    Args:
        path: Path to csv file of dataset.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.

    Returns:
        (contexts: list, questions: list, answers: List[Dict]).
        Format of answers dict: answers[0] = {'text': 'some answer'}.

    """
    df = pd.read_csv(path)

    i_start = int((part-1) / parts_num * len(df))
    i_finish = int(part / parts_num * len(df))

    contexts = df.text[i_start:i_finish].to_list()
    questions = df.question[i_start:i_finish].to_list()
    answers = df.span_answer[i_start:i_finish].apply(lambda x: {'text': x}).to_list()

    return contexts, questions, answers


def add_idx(answers: List[dict], contexts: List[str]) -> NoReturn:
    """Appends start and end indices of cpan answer of the context.

    Indices are appended to answer dicts. In case of answer absence,
    start and end indices are equal to 0.

    Args:
        answers: List of span answers
        contexts: List of contexts.

    """
    for answer, context in zip(answers, contexts):
        i = 0
        if type(answer['text']) == str:
            gold_text = answer['text']
            start_idx = context.find(gold_text)
            assert start_idx != -1, f"Answer {i} is  not found"
            end_idx = start_idx + len(gold_text)

            answer['answer_start'] = start_idx
            answer['answer_end'] = end_idx
        else:
            answer['answer_start'] = 0
            answer['answer_end'] = 0
        i += 1


def add_token_positions(encodings, answers, tokenizer):
    """Adds start and end positions of the answers int encodings.

    Args:
        encodings: Tokenizer encodings.
        answers (List(dict): Span answers.
        tokenizer: Huggingface tokenizer.

    Returns:
        Updated encodings with start and end positions.

    """
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        #no answer case
        if answers[i]['answer_start'] == 0 and answers[i]['answer_end'] == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def prepare_data(csv_path, tokenizer, parts_num = 1, part = 1):
    """Prepares torch dataset from csv file.

    The dataset must contain columns: 'text', 'question', 'span_answer'.

    Args:
        csv_path: Path to scv file.
        tokenizer: Huggingface tokenizer.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.

    Returns:
        torch.dataset: prepared dataset.

    """
    contexts, questions, answers = read_data(csv_path, parts_num, part)
    add_idx(answers, contexts)
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers, tokenizer)
    dataset = TrainDataset(encodings)

    return dataset


def read_and_merge_squad_data(path, text_num=1, parts_num=1, part=1, text_ids=None):
    """Divides dataframe into context, question and answer lists.

    Function can prepare as whole dataset as only its part.

    The dataset must contain columns: 'question', 'answer'.

    Also dataset must contain one or few text columns that can be merged.
    If dataset has only one text column, it must be named 'context'.
    In case of few columns its name must be as follows: 't1', 't2', ...

    Args:
        path: Path to csv file of dataset.
        text_num: Number of text columns in the dataset.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.
        text_ids: Text ids that will be merged.

    Returns:
        (contexts: list, questions: list, answers: List[Dict]).
        Format of answers dict: answers[0] = {'text': 'some answer'}.

    """
    df = pd.read_csv(path)

    i_start = int((part-1) / parts_num * len(df))
    i_finish = int(part / parts_num * len(df))

    
    questions = df.question[i_start:i_finish].to_list()
    answers = df.answer[i_start:i_finish].apply(lambda x: {'text': x}).to_list()

    if text_num != 1:
        if text_ids:
            if len(text_ids) == 1:
                contexts = df[f"t{text_ids[0]}"][i_start:i_finish].to_list()
            else:
                text_col_names = [f"t{i}" for i in text_ids]
                texts = df.loc[i_start:i_finish, text_col_names].to_numpy()
                contexts = [' '.join(item) for item in texts]
        else:    
            text_col_names = [f"t{i+1}" for i in range(text_num)]
            texts = df.loc[i_start:i_finish, text_col_names].to_numpy()
            contexts = [' '.join(item) for item in texts]
    else:
        contexts = df.context[i_start:i_finish].to_list()

    return contexts, questions, answers


def read_and_merge_msmarco_data(path, text_num=1, parts_num = 1, part = 1, text_ids = None):
    """Divides dataframe into context, question and answer lists.

    Function can prepare as whole dataset as only its part.

    The dataset must contain columns: 'question', 'span_answer'.

    Also dataset must contain one or few text columns that can be merged.
    If dataset has only one text column, it must be named 'text'.
    In case of few columns its name must be as follows: 't1', 't2', ...

    Args:
        path: Path to csv file of dataset.
        text_num: Number of text columns in the dataset.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.
        text_ids: Text ids that will be merged.

    Returns:
        (contexts: list, questions: list, answers: List[Dict]).
        Format of answers dict: answers[0] = {'text': 'some answer'}.

    """
    df = pd.read_csv(path)

    i_start = int((part-1) / parts_num * len(df))
    i_finish = int(part / parts_num * len(df))

    
    questions = df.question[i_start:i_finish].to_list()
    answers = df.span_answer[i_start:i_finish].apply(lambda x: {'text': x}).to_list()

    if text_num != 1:
        if text_ids:
            if len(text_ids) == 1:
                contexts = df[f"t{text_ids[0]}"][i_start:i_finish].to_list()
            else:
                text_col_names = [f"t{i}" for i in text_ids]
                texts = df.loc[i_start:i_finish, text_col_names].to_numpy()
                contexts = [' '.join(item) for item in texts]
        else:    
            text_col_names = [f"t{i+1}" for i in range(text_num)]
            texts = df.loc[i_start:i_finish, text_col_names].to_numpy()
            contexts = [' '.join(item) for item in texts]
    else:
        contexts = df.text[i_start:i_finish].to_list()

    return contexts, questions, answers


def prepare_and_merge_data(csv_path, tokenizer, data_type, text_num=1, parts_num=1,
                           part=1,  text_ids=None, return_answers=False):
    """Prepares torch dataset from csv file.

    Function can merge few text columns, and prepares QAT triplets
    in form torch.dataset.

    Args:
        csv_path: Path to dataset.
        tokenizer: Huggingface tokenizer.
        data_type: Type of the dateaset, must be 'squad' or 'msmarco'.
            In case of squad, dataset should contain 'question', 'span_answer', 'context'(or 't1', 't2', ...) columns.
            In case of msmarco, dataset should contain 'question', 'answer', 'text'(or 't1', 't2', ...) columns.
        text_num: Number of text columns in the dataset.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.
        text_ids: Text ids that will be merged.
        return_answers: Either return text answers or not.

    Returns:
        If return_answers = False, returns torch dataset.
        If return_answers = True, returns tuple (torch dataset, list of text answers).

    """
    assert data_type in ['squad', 'msmarco'], "Data type should be 'squad' or 'msmarco'."

    if data_type == 'squad':
        contexts, questions, answers = read_and_merge_squad_data(csv_path, text_num, parts_num, part, text_ids)
    if data_type == 'msmarco':
        contexts, questions, answers = read_and_merge_msmarco_data(csv_path, text_num, parts_num, part, text_ids)

    add_idx(answers, contexts)
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers, tokenizer)
    dataset = TrainDataset(encodings)
    if return_answers:
        return dataset, [ans['text'] for ans in answers]
    else:
        return dataset


def prepare_and_merge_eval_data(csv_path, tokenizer, text_num=1, parts_num=1, part=1, text_ids=None):
    """Function which is pass do dataset only nan answers.

    Function can merge few text columns, and prepares QAT triplets
    in form torch.dataset.

    Dataset must contain 'question', 'span_answer', 'context'(or 't1', 't2', ...) columns.

    Args:
        csv_path: Path to dataset.
        tokenizer: Huggingface tokenizer.
        text_num: Number of text columns in the dataset.
        parts_num: The number of parts the dataset will be divided into.
        part: Part number, that wil be returned.
        text_ids: Text ids that will be merged.

    Returns:
        torch.dataset: Prepared torch dataset.

    """
    contexts, questions, answers = read_and_merge_squad_data(csv_path, text_num, parts_num, part, text_ids)
    answers = [{'text': float('nan')} for i in range(len(answers))]
    add_idx(answers, contexts)
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers, tokenizer)
    dataset = TrainDataset(encodings)
    return dataset

"""# Metrics computation"""

#new fanction for f1 calculation
def return_text_results(bert_results, input_dataset, tokenizer):
    def select_answer(start_logits, end_logits, id, input_dataset, top_n=20):
        start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
  
        tokens = input_dataset.__getitem__(id)['input_ids']
        sep = tokens.tolist().index(102)
      
        prelim_preds = []
        while not prelim_preds:
            start_indexes = [idx for idx, logit in start_idx_and_logit[:top_n]]
            end_indexes = [idx for idx, logit in end_idx_and_logit[:top_n]]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # throw out invalid predictions
                    if start_index not in range(1, sep) or end_index not in range(1, sep):
                      continue
                    if end_index < start_index:
                        continue
                    prelim_preds.append([start_index, end_index, start_logits[start_index] + end_logits[end_index]])

            top_n = top_n * 2

        prelim_preds.append([0, 0, start_logits[0] + end_logits[0]])
        prelim_preds = sorted(prelim_preds, key=lambda x: x[2], reverse=True)

        #extract text from tokens  
        pred_start, pred_finish = prelim_preds[0][:2]

        if pred_start == 0:
            return '', prelim_preds[0][2]
        else:
            text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(
                    tokens[pred_start:pred_finish+1]
                )
            )

            return text, prelim_preds[0][2] 

    #beginning of the function implementation
    start_logits = bert_results.predictions[0]
    finish_logits = bert_results.predictions[1]

    text_and_score = []
    id = 0
    for start_logit, finish_logit in zip(start_logits, finish_logits):
        text_and_score.append(select_answer(start_logit, finish_logit, id, input_dataset, top_n=10))
        id += 1
    
    return text_and_score

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_score_squad(trainer, answers, tokenizer, eval_dataset = None):
    if eval_dataset is None:
        eval_dataset = trainer.eval_dataset

    predictions = trainer.predict(eval_dataset)
    pred_texts = return_text_results(predictions, eval_dataset, tokenizer)

    f1 = []
    for pred_text, answer in zip(pred_texts, answers):
        pred_text = pred_text[0]
        if type(answer) != str:
            answer = ''
        f1.append(compute_f1(pred_text, answer))

    return {'f1_mean': np.mean(f1)}

#old functions for metrics calculation
def compute_f1_f(predict, eval_dataset):

    def select_answer(start_logits, end_logits, id, top_n=20):

        start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
        end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
  
        tokens = eval_dataset.__getitem__(id)['input_ids']
        sep = tokens.tolist().index(102)
      
        prelim_preds = []
        while not prelim_preds:
            start_indexes = [idx for idx, logit in start_idx_and_logit[:top_n]]
            end_indexes = [idx for idx, logit in end_idx_and_logit[:top_n]]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # throw out invalid predictions
                    if start_index not in range(1, sep) or end_index not in range(1, sep):
                      continue
                    if end_index < start_index:
                        continue
                    prelim_preds.append([start_index, end_index, start_logits[start_index] + end_logits[end_index]])

            top_n = top_n * 2

        prelim_preds.append([0, 0, start_logits[0] + end_logits[0]])
        prelim_preds = sorted(prelim_preds, key=lambda x: x[2], reverse=True)
        return prelim_preds[0][:2]  
    
    def return_f(true_start, pred_start, true_finish, pred_finish):
      len_pred_tokens = np.maximum(0, pred_finish-pred_start)
      len_true_tokens = np.maximum(0, true_finish-true_start)
      # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
      if len_pred_tokens == 0 or len_true_tokens == 0 or pred_start**2 + (pred_finish-1)**2 == 0 or true_start**2 + (true_finish-1)**2 == 0 :
          return int(true_start == pred_start and true_finish==pred_finish)
    
      common_tokens_len = max(0, min(true_finish, pred_finish) - max(true_start, pred_start))
      
      # if there are no common tokens then f1 = 0
      if common_tokens_len == 0:
          return 0
      
      prec = common_tokens_len / len_pred_tokens
      rec = common_tokens_len / len_true_tokens
      
      return 2 * (prec * rec) / (prec + rec)

    true_start_id = predict.label_ids[0]
    true_finish_id = predict.label_ids[1]+1
    start_logits = predict.predictions[0]
    finish_logits = predict.predictions[1]
    
    f1 = []
    id = 0
    for true_start, true_finish, start_logit, finish_logit in zip(true_start_id, true_finish_id, start_logits, finish_logits):
      pred_start, pred_finish = select_answer(start_logit, finish_logit, id)
      id += 1
      pred_finish += 1
      f1.append(return_f(true_start, pred_start, true_finish, pred_finish))
    
    f1 = np.array(f1)

    return {"f1_score": 100 * f1.mean(),
            "Exact Match": 100 * (f1 == 1).mean(),
            "f1_list": 100 * f1}

def evaluate_score(trainer, eval_dataset = None):
    if eval_dataset is None:
        eval_dataset = trainer.eval_dataset

    pred = trainer.predict(eval_dataset)
    data = compute_f1_f(pred, eval_dataset)

    #with open(trainer.args.output_dir + '/evaluation_results.json', 'w') as outfile:
     #   json.dump(data, outfile)

    return data