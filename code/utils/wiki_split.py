import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, NoReturn, Any, Optional, Union

def context_split(context:str, stride:int = 128, split_length:int = 512, drop_rate:float = 0.5) -> List[str]: # return list (stride shout be bigger than 83)
    """
    Summary:
        split given context into segments for a whole context to be used in similarity ranking; not to be lost while tokenizing a context.

    Args:
        context (str): full context.
        stride (int, optional): overlap section to avoid losing answer while spliting given context. Defaults to 128.
        split_length (int, optional): length of splitted segments. Defaults to 512.
        drop_rate (float, optional): drop rate for minimum text length. Defaults to 0.5.   
                                        e.g. split_length=512, drop_rate=0.5 -> 256 is the minimum length of text segment. If segment is shorter than 256, then drop.

    Returns:
        context_split_list (List[str]): context segments list.
    """

    context_split_list = []

    for start_idx in range(0, len(context), split_length-stride):
        end_idx = start_idx + split_length if start_idx + split_length <= len(context) else len(context)
        if int(split_length * drop_rate) < (end_idx - start_idx):
            context_split_list.append(context[start_idx:end_idx])
    
    return context_split_list

def main(args):
    new_json = dict()
    stride, split_length, drop_rate = 128, 512, 0.5

    # load wiki data.
    wiki_json = None
    with open(os.path.join(args.data_dir, args.data_file_name), 'r', encoding='utf-8') as f:
        wiki_json = json.load(f)

    # json convert to pandas DataFrame for convenience.
    wiki_df = pd.DataFrame.from_dict(wiki_json, orient='index')
    wiki_df = wiki_df.drop_duplicates(['text'], keep='first')

    # iterate over pandas DataFrame rows.
    for _, wiki in tqdm(wiki_df.iterrows(), desc="Making New Wiki Dataset..."):
        # text length & drop rate check (description is mentioned above)
        if len(wiki['text']) > int(split_length * (1+drop_rate)):
            text_list = context_split(wiki['text'], stride=stride, split_length=split_length, drop_rate=drop_rate)

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

    # save new wiki data.
    with open(os.path.join(args.data_dir, args.save_file_name), 'w', encoding='utf-8') as f:
        json.dump(new_json, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../data',
        help="data directory (default: ../../data)"
    )
    parser.add_argument(
        '--data_file_name',
        type=str,
        default='wikipedia_documents.json',
        help="data file name (default: wikipedia_documents.json)"
    )
    parser.add_argument(
        '--save_file_name',
        type=str,
        default='wiki_split.json',
        help="save file name (default: wiki_split.json)"
    )

    args = parser.parse_args()

    main(args)