import json
import os
import argparse
from typing import Dict, List, NoReturn, Optional, Tuple
from simple_parsing import ArgumentParser

from arguments import EnsembleArguments

from collections import (
    defaultdict, 
    OrderedDict,
    Counter
)

def _load_nbest_data(nbest_dir: str) -> Tuple[List[Dict[str, List[Dict]]], List[str]]:
    """
    Summary:
        load nbest prediction files.

    Args:
        nbest_dir (str): nbest prediction files directory.

    Returns:
        nbest_json, id_list (Tuple[List[Dict[str, List[Dict]]], List[str]]): nbest prediction files converted to dict list, and key id list.
    """
    assert os.path.exists(nbest_dir), "NBEST FILES DIRECTORY DOES NOT EXIST."
    assert len(os.listdir(nbest_dir)) > 2, "THE NUMBER OF NBEST FILES SHOULD BE MORE THAN 1."

    nbest_json = []
    for nbest_file in sorted(os.listdir(nbest_dir)):
        with open(f"{nbest_dir}/{nbest_file}", "r", encoding="utf-8") as f:
            nbest_json.append(json.load(f))
            
    print(f"NBEST FILES COUNT : {len(nbest_json)}")

    id_list = []
    for id_ in nbest_json[0].keys():
        id_list.append(id_)
    
    return nbest_json, id_list

def _save_ensemble(prediction: OrderedDict, output_dir: str, postfix: Optional[str] = None) -> NoReturn:
    """
    Summary:
        save prediction result from ensemble into json.

    Args:
        prediction (OrderedDict): prediction result from ensemble.
        output_dir (str): saving directory.
        postfix (Optional[str]): postfix of output file name. Defaults to None.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("ERROR: FAILED TO CREATE OUTPUT DIECTRORY")
    
    if postfix:
        with open(f"{output_dir}/predictions_{postfix}.json", "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(prediction, indent=4, ensure_ascii=False) + "\n"
            )
    else:
        with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(prediction, indent=4, ensure_ascii=False) + "\n"
            )

def _hard_voting(nbest_json: List[Dict[str, List[Dict]]], id_list: List[str]) -> OrderedDict:
    """
    Summary:
        ensemble with hard voting.

    Args:
        nbest_json (List[Dict[str, List[Dict]]]): nbest prediction files converted to dict list.
        id_list (List[str]): key id list.

    Returns:
        prediction (OrderedDict): prediction result from hard voting.
    """
    prediction = OrderedDict()

    for id_ in id_list:
        hit_count_dict = defaultdict(int)

        for nbest in nbest_json:
            gold_text = nbest[id_][0]['text']
            hit_count_dict[gold_text] += 1
    
        counter = Counter(hit_count_dict).most_common()
        prediction[id_] = counter[0][0]
    
    return prediction

def _soft_voting(nbest_json: List[Dict[str, List[Dict]]], id_list: List[str]) -> OrderedDict:
    """
    Summary:
        ensemble with soft voting.

    Args:
        nbest_json (List[Dict[str, List[Dict]]]): nbest prediction files converted to dict list.
        id_list (List[str]): key id list.

    Returns:
        prediction (OrderedDict): prediction result from soft voting.
    """
    prediction = OrderedDict()

    for id_ in id_list:
        probs_dict = defaultdict(int)

        for nbest in nbest_json:
            for _, candidate in enumerate(nbest[id_]):
                probs_dict[candidate['text']] += candidate['probability']
    
        hit = max([(probs/len(nbest_json), text) for text, probs in probs_dict.items()])
        prediction[id_] = hit[1]

    return prediction

def main(args):
    assert (args.do_hard_voting or args.do_soft_voting), "SELECT AT LEAST ONE ENSEMBLE TYPE."
    nbest_json, id_list = _load_nbest_data(args.nbest_dir)

    hard_voting_prediction, soft_voting_prediction = None, None
        
    if args.do_hard_voting:
        print(f"Ensemble Start... You are doing HARD VOTING")
        hard_voting_prediction = _hard_voting(nbest_json, id_list)

        print("Saving Prediction...")
        _save_ensemble(
            prediction=hard_voting_prediction, 
            output_dir=args.output_dir,
            postfix="hard"
        )

    if args.do_soft_voting:
        print(f"Ensemble Start... You are doing SOFT VOTING")
        soft_voting_prediction = _soft_voting(nbest_json, id_list)

        print("Saving Prediction...")
        _save_ensemble(
            prediction=soft_voting_prediction, 
            output_dir=args.output_dir,
            postfix="soft"
        )
    
    print("Ensemble Finished!")

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_arguments(EnsembleArguments, dest="args")
    parser = arg_parser.parse_args()

    print("==================== ENSEMBLE ARGUMENTS ====================")
    for arg in vars(parser.args):
        print(f"{arg} : {getattr(parser.args, arg)}")
    print("============================================================")
    
    main(parser.args)