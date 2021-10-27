import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, NoReturn, Any, Optional, Union

def document_split(text:str, stride:int = 83, split_length:int = 512, drop_rate:float = 0.5) -> List[str]: # return list (stride shout be bigger than 83)
    """
    Summary:
        split given document into segments.

    Args:
        text (str): full text.
        stride (int, optional): overlap section to avoid losing answer while spliting document. Defaults to 83 that is max length of train data answer.
        split_length (int, optional): length of splitted segments. Defaults to 512.
        drop_rate (float, optional): drop rate for minimum text length. Defaults to 0.5.   
                                        ex) split_length:512, drop_rate:0.5 -> 256 is the minimum length of text segment. If segment is shorter than 256, then drop

    Returns:
        List[str]: text segments list.
    """
    text_split_list = []

    for start_idx in range(0, len(text), split_length-stride):
        end_idx = start_idx + split_length if start_idx + split_length <= len(text) else len(text)
        if int(split_length * drop_rate) < (end_idx - start_idx):
            text_split_list.append(text[start_idx:end_idx])
    
    return text_split_list

def main():
    new_json = dict()
    stride, split_length, drop_rate = 128, 512, 0.5

    # load wiki data as json.
    wiki_json = None
    with open('../data/wikipedia_documents.json', 'r', encoding='utf-8') as f:
        wiki_json = json.load(f)

    # json convert to pandas DataFrame for convenience.
    wiki_df = pd.DataFrame.from_dict(wiki_json, orient='index')
    wiki_df = wiki_df.drop_duplicates(['text'], keep='first', ignore_index=True)

    # iterate over pandas DataFrame rows.
    for _, wiki in tqdm(wiki_df.iterrows(), desc="Making New Wiki Dataset..."):
        # text length & drop rate check (description is mentioned above)
        if len(wiki['text']) > int(split_length * (1+drop_rate)):
            text_list = document_split(wiki['text'], stride=stride, split_length=split_length, drop_rate=drop_rate)

            # add splitted text segments to new wiki dict.
            for idx, text in enumerate(text_list, start=1):
                new_json[str(wiki['document_id']) + f'_{idx}'] = {
                    'text': text,
                    'title': wiki['title'],
                    'document_id': str(wiki['document_id']) + f'_{idx}'
                }
        else:
            new_json[str(wiki['document_id'])] = {
                'text': wiki['text'],
                'title': wiki['title'],
                'document_id': str(wiki['document_id'])
            }

    # new wiki dict to json.
    with open('../data/wiki_split.json', 'w') as f:
        json.dump(new_json, f)
    
if __name__ == '__main__':
    main()