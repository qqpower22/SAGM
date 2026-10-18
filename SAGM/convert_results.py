import json
import numpy as np
import pandas as pd


def remove_special_tokens(preds):
    for pred in preds:
        assert '[POS]' == pred['sentence'][0]
        pred['sentence'] = pred['sentence'][2:]
        for ent in pred['entity']:
            ent['index'] = [x - 2 for x in ent['index']]
    return preds
