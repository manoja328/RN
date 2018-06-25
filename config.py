import os
dataset = {}
dataset['Ncls'] = 28
CLEVR = '/home/manoj/CLEVR_v1.0/'
dataset['CLEVR'] = CLEVR
dataset['train'] = CLEVR + 'questions/CLEVR_train_questions.json'
dataset['test'] =  CLEVR + 'questions/CLEVR_test_questions.json'
dataset['val'] =   CLEVR + 'questions/CLEVR_val_questions.json'
