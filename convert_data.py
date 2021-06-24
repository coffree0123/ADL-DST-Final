import os
import json
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from copy import deepcopy


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def dump_json(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4)


def create_slot(slots):
    return_slot = {}
    for slot_name in slots.keys():
        return_slot[slot_name.replace("_", "-")] = slots[slot_name][0]
    return return_slot


def turns2dialogue(user_turn: dict, sys_turn: dict) -> dict:
    assert user_turn['speaker'] == 'USER' and (
        not sys_turn or sys_turn['speaker'] == 'SYSTEM')

    frame = user_turn['frames'][0]  # only consider frame_0

    dialogue = {}
    dialogue['system'] = sys_turn['utterance'] if sys_turn else "none"
    dialogue['user'] = user_turn['utterance']
    dialogue['state'] = {"active_intent": "none",
                         "slot_values": create_slot(frame["state"]["slot_values"])}

    return dialogue


def convert_one_entity(ent: dict, ontology: dict, is_test=False) -> dict:
    res, belief_state = {}, {}
    belief_state["active_intent"] = "none"
    belief_state["slot_values"] = {}
    # dialogue_id -> dialogue_idx
    res['dial_id'] = ent['dialogue_id']
    # services -> domains
    res['domains'] = [service.replace("_", "-") for service in ent['services']]
    # turns -> dialogue
    res['turns'] = []
    sys_turn = None
    for idx, turn in enumerate(ent['turns']):
        if (is_test):
            if idx % 2 == 0:
                user_turn = turn
                dialogue = {}
                dialogue['system'] = sys_turn['utterance'] if sys_turn else "none"
                dialogue['user'] = user_turn['utterance']
                dialogue['state'] = belief_state
                res['turns'].append(dialogue)
            else:
                sys_turn = turn
        else:
            if idx % 2 == 0:
                user_turn = turn
                try:
                    dialogue = turns2dialogue(user_turn, sys_turn)
                    res['turns'].append(dialogue)
                except:
                    pass
            else:
                sys_turn = turn

    return res


def convert(data: list, ontology: dict, is_test=False) -> list:
    res = [convert_one_entity(d, ontology, is_test) for d in data]
    return res


def create_ontology(schema_path, ontology):
    domains = {}
    return_des, return_onto = {}, {}
    schema = load_json(schema_path)
    for service in schema:

        service_name = service["service_name"].replace("_", "-")
        if (service_name not in domains):
            domains[service_name] = 0
        domains[service_name] += 1

        slots = service["slots"]
        for slot in slots:
            if (slot.get("possible_values", -1) == -1):
                print(slot["name"])
                continue

            slot_name = slot["name"].replace('_', '-')
            # Create slot_description.json
            return_des[slot_name] = {
                "description_human": slot["description"], "values": slot["possible_values"]}

            # Create new ontology
            return_onto[slot_name] = slot["possible_values"]

    return return_des, return_onto, domains


def main(args):

    ontology = {}

    src_dir = Path(args.src_dir)

    output_dir = os.path.join(args.output_dir, "data")

    print(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_path = Path(os.path.join(src_dir, "train"))

    dev_path = Path(os.path.join(src_dir, "dev"))

    # For training data.
    result = []

    src_files = sorted(
        train_path.glob('*.json'),
        key=lambda x: int(str(x).split('.')[0].split('_')[1])
    )

    for src_file in tqdm(src_files):
        src_data = load_json(src_file)
        result += convert(src_data, ontology)

    dump_json(result, os.path.join(output_dir, "train_dials.json"))

    # For validatin data.
    result = []

    src_files = sorted(
        dev_path.glob('*.json'),
        key=lambda x: int(str(x).split('.')[0].split('_')[1])
    )

    for src_file in tqdm(src_files):
        src_data = load_json(src_file)
        result += convert(src_data, ontology)

    dump_json(result, os.path.join(output_dir, "dev_dials.json"))

    # Create test seen data.
    test_path = Path(os.path.join(src_dir, "test_seen"))

    result = []

    src_files = sorted(
        test_path.glob('*.json'),
        # test"_"seen/dialogues"_"001.json.
        key=lambda x: int(str(x).split('.')[0].split('_')[2])
    )

    for src_file in tqdm(src_files):
        src_data = load_json(src_file)
        result += convert(src_data, ontology, is_test=True)

    dump_json(result, os.path.join(output_dir, "test_seen_dials.json"))

    # Create test unseen data.
    test_path = Path(os.path.join(src_dir, "test_unseen"))

    result = []

    src_files = sorted(
        test_path.glob('*.json'),
        # test"_"seen/dialogues"_"001.json.
        key=lambda x: int(str(x).split('.')[0].split('_')[2])
    )

    for src_file in tqdm(src_files):
        src_data = load_json(src_file)
        result += convert(src_data, ontology, is_test=True)

    dump_json(result, os.path.join(output_dir, "test_unseen_dials.json"))

    schema_path = os.path.join(src_dir, "schema.json")

    schema, ontology, domains = create_ontology(schema_path, ontology)

    dump_json(ontology, os.path.join(output_dir, "ontology.json"))

    dump_json(schema, os.path.join(args.output_dir,
              "utils", "slot_description.json"))

    print(domains.keys())


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--src_dir", help="source data dir")
    parser.add_argument("-o", "--output_dir", help="output directory")
    args = parser.parse_args()

    main(args)
