import json
import numpy as np
import pandas as pd
import os
import nltk
path = ['data','paragraphs','qas','answers']
curr_dir = os.getcwd()
def get_data_frame():
    js = pd.io.json.json_normalize(json.loads(open(curr_dir+'/train-v1.1.json').read()),path)
    m = pd.io.json.json_normalize(json.loads(open(curr_dir+'/train-v1.1.json').read()),path[:-1])
    r = pd.io.json.json_normalize(json.loads(open(curr_dir+'/train-v1.1.json').read()),path[:-2])
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id','question', 'context']].set_index('id'),js.set_index('q_idx')],1).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    return main
    