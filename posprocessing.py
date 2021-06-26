import os
import sys
import json


def json_load(path):

    with open(path) as json_file:
        data = json.load(json_file)

    return data


def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                        slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))

def check_begin(slot):
    is_begin = False
    begin_word = ["origin", "departure"]
    for word in begin_word:
        if slot.find(word) != -1:
            is_begin = True
    return is_begin


def get_slot_values(turns: dict) -> dict:
    ans = {}
    for k, v in turns.items():
        pred_belief = v["pred_belief"]
        for state in pred_belief:
            slot_name, slot_value = state.split("#####")[0], state.split("#####")[1]
            if (not check_begin(slot_name) or ans.get(slot_name, "none") == "none"):
                ans[slot_name] = slot_value

    return ans


if __name__ == '__main__':
    prediction = {}

    path = sys.argv[1]

    predict = json_load(path)

    for dial_id in predict.keys():

        dialogue = predict[dial_id]

        prediction[dial_id] = get_slot_values(dialogue["turns"])

    write_csv(prediction, sys.argv[2])
