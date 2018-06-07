#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:54:55 2018

@author: manoj
"""
import torch
from collections import Counter
import itertools
from tqdm import tqdm
import re
import json
import argparse      
import os
# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


ANS = ['0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 '10',
 'blue',
 'brown',
 'purple',
 'cyan',
 'gray',
 'green',
 'yellow',
 'red',
 'cube',
 'cylinder',
 'sphere',
 'large',
 'small',
 'metal',
 'rubber',
 'yes',
 'no',
  ]


ANS_to_idx = {ans:idx for ans,idx in zip(ANS,range(len(ANS)))}
idx_to_ANS = {idx:ans for ans,idx in zip(ANS,range(len(ANS)))}


class Lang:
  
    token_to_index =  None   
    answer_to_index =  None   
   
    def __init__(self,max_question_length=43):
    
        self.max_question_length = max_question_length 
        
        if os.path.exists("vocab.json"):
            print ("loading from saved file....")
            with open('vocab.json','r') as f:
                vocab = json.load(f)
                self.token_to_index = vocab['question']
                self.answer_to_index = ANS_to_idx 
        else:
            print ("vocab file not found !!")
            print ("run python lang.py")
        
 
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0    
    
    @staticmethod
    def extract_vocab(iterable, top_k=None, start=0):
        """ Turns an iterable of list of tokens into a vocabulary.
            These tokens could be single answers or word tokens in questions.
        """
        all_tokens = itertools.chain.from_iterable(iterable)
        counter = Counter(all_tokens)
        if top_k:
            most_common = counter.most_common(top_k)
            most_common = (t for t, c in most_common)
        else:
            most_common = counter.keys()   
            
        most_common = counter.keys()
        # descending in count, then lexicographical order
        tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
        vocab = {t: i for i, t in enumerate(tokens, start=start)}
        return vocab


    def prepare_question(self,question):
        """ Tokenize and normalize questions from a given question json in the usual VQA format. """       
        question = question.lower()[:-1]          
        return question.split(' ')

    def encode_question(self,question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()        
        question_tokens = self.prepare_question(question)
        for i, token in enumerate(question_tokens):
            index = self.token_to_index.get(token, 0)
            if i >=self.max_question_length:
                break
            vec[i] = index
        L = len(question_tokens)
        if L>self.max_question_length:
            L = self.max_question_length
        return vec, L


    def encode_answer(self, answer):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
                
        index = self.answer_to_index.get(answer)
        return index


    def prepare_answers(self,answers):
        """ Normalize answers from a given answer json in the usual VQA format. """
        # The only normalization that is applied to both machine generated answers as well as
        # ground truth answers is replacing most punctuation with space (see [0] and [1]).
        # Since potential machine generated answers are just taken from most common answers, applying the other
        # normalizations is not needed, assuming that the human answers are already normalized.
        # [0]: http://visualqa.org/evaluation.html
        # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
    
        def process_punctuation(s):
            # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
            # this version should be faster since we use re instead of repeated operations on str's
            if _punctuation.search(s) is None:
                return s
            s = _punctuation_with_a_space.sub('', s)
            if re.search(_comma_strip, s) is not None:
                s = s.replace(',', '')
            s = _punctuation.sub(' ', s)
            s = _period_strip.sub('', s)
            return s.strip()
    
        answers = set(answers)
        return [[process_punctuation(ans)] for ans in answers]

#%%
            
def main(data=None,top_k=100 , save=False):
    
    lang = Lang()
    questions_tok= []
    print ("Using {} answers".format(top_k))
    assert data!=None, "data cannot be None!!!"
    questions = [q['question'] for q in data] 
    
    for question in tqdm(questions):
        qtokens = lang.prepare_question(question)
        questions_tok.append(qtokens)
    
    token_to_index = lang.extract_vocab(questions_tok, start=1)           
    answers = [q['answer'] for q in data] 
    answers = lang.prepare_answers(answers)
    answer_to_index = lang.extract_vocab(answers,top_k=top_k)            
            
    if save:
        vocabs = {
            'question': token_to_index,
            'answer': answer_to_index,
        }
        with open('vocab.json', 'w') as fd:
            json.dump(vocabs, fd)
                
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=list,help='List of dicts with question as field',default=None)
    parser.add_argument('--nans', type=int,help='Number of answers',default=28)
    parser.add_argument('--save', help='save',type=bool,default=True)
    args = parser.parse_args()
    print (args)
    assert args.data!=None, "data cannot be None!!!"
    main(args.data,args.nans,args.save)
    