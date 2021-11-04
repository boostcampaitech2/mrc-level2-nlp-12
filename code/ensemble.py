import json
import os
import argparse
from typing import Dict, List, NoReturn, Optional, Tuple
from simple_parsing import ArgumentParser

from arguments import EnsembleArguments

from collections import defaultdict, OrderedDict, Counter


def __load_nbest_data(nbest_dir: str) -> Tuple[List[Dict[str, List[Dict]]], List[str]]:
    assert os.path.exists(nbest_dir), "NBEST DIRECTORY DOES NOT EXIST."
    assert len(os.listdir(nbest_dir)) > 2, "NBEST DATA FILE SHOULD BE MORE THAN 1."

    nbest_json = []
    for nbest_file in sorted(os.listdir(nbest_dir)):
        with open(f"{nbest_dir}/{nbest_file}", "r", encoding="utf-8") as f:
            nbest_json.append(json.load(f))

    print(f"NBEST COUNT : {len(nbest_json)}")

    id_list = []
    for id_ in nbest_json[0].keys():
        id_list.append(id_)

    return nbest_json, id_list


def __save_ensemble(
    prediction: OrderedDict, output_dir: str, postfix: Optional[str] = None
) -> NoReturn:
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print("ERROR: FAILED TO CREATE OUTPUT DIECTRORY")

    if postfix:
        with open(
            f"{output_dir}/predictions_{postfix}.json", "w", encoding="utf-8"
        ) as writer:
            writer.write(json.dumps(prediction, indent=4, ensure_ascii=False) + "\n")
    else:
        with open(f"{output_dir}/predictions.json", "w", encoding="utf-8") as writer:
            writer.write(json.dumps(prediction, indent=4, ensure_ascii=False) + "\n")


def __hard_voting(
    nbest_json: List[Dict[str, List[Dict]]], id_list: List[str]
) -> OrderedDict:
    prediction = OrderedDict()

    for id_ in id_list:
        hit_count_dict = defaultdict(int)
        probs_dict = defaultdict(int)

        for nbest in nbest_json:
            temp_dict = defaultdict(int)

            for candidate in nbest[id_]:
                temp_dict[candidate["text"]] += candidate["probability"]

            gold_text = max(temp_dict, key=temp_dict.get)
            hit_count_dict[gold_text] += 1
            probs_dict[gold_text] += temp_dict[gold_text]

        counter = Counter(hit_count_dict).most_common()
        if len(counter) > 1 and counter[0][1] > counter[1][1]:  # hard voting
            prediction[id_] = counter[0][0]
        else:  # unable to do hard voting, get max prob text instead.
            prediction[id_] = max(probs_dict, key=probs_dict.get)

    return prediction


def __soft_voting(
    nbest_json: List[Dict[str, List[Dict]]], id_list: List[str]
) -> OrderedDict:
    prediction = OrderedDict()

    for id_ in id_list:
        probs_dict = defaultdict(int)

        for nbest in nbest_json:
            for candidate in nbest[id_]:
                probs_dict[candidate["text"]] += candidate["probability"]

        hit = max(
            [(probs / len(nbest_json), text) for text, probs in probs_dict.items()]
        )
        prediction[id_] = hit[1]

    return prediction


def main(args):
    assert (
        args.do_hard_voting or args.do_soft_voting
    ), "SELECT AT LEAST ONE ENSEMBLE TYPE."
    nbest_json, id_list = __load_nbest_data(args.nbest_dir)

    hard_voting_prediction, soft_voting_prediction = None, None

    if args.do_hard_voting:
        print(f"Ensemble Start... You are doing HARD VOTING")
        hard_voting_prediction = __hard_voting(nbest_json, id_list)

        print("Saving Prediction...")
        __save_ensemble(
            prediction=hard_voting_prediction,
            output_dir=args.output_dir,
            postfix="hard",
        )

    if args.do_soft_voting:
        print(f"Ensemble Start... You are doing SOFT VOTING")
        soft_voting_prediction = __soft_voting(nbest_json, id_list)

        print("Saving Prediction...")
        __save_ensemble(
            prediction=soft_voting_prediction,
            output_dir=args.output_dir,
            postfix="soft",
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
