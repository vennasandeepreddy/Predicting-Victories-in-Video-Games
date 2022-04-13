import json
import pickle
from os import walk

def get_binary_encoding(num):
    res = []
    for i in range(3):
        res.append(num % 2)
        num //= 2
    return list(reversed(res))

unitTypeMap = {"Assault": 0,
               "Heavy": 1,
               "Scout": 2,
               "Support": 3,
               "Captain": 4}


def get_features_from_state(state, prevTimeStepFeatures):
    features = prevTimeStepFeatures.copy()
    if 'hp' in state:
        features[1] = round(state['hp']['current'] / state['hp']['max'], 2)
    if 'ap' in state:
        features[0] = round(state['ap']['current'] / state['ap']['max'], 2)
    return features


def update_dataset_with_new_state(dataset, state):
    if "hp" in state or 'ap':
        if state['id'] in dataSet[game_id][1]:
            f_new = get_features_from_state(state, dataSet[game_id][1][state['id']][-1])
            if f_new == dataSet[game_id][1][state['id']][-1]:
                # The states are same as previous state.
                return
            for playerId in dataSet[game_id][1]:
                if state['id'] == playerId:
                    dataSet[game_id][1][playerId].append(f_new)
                else:
                    dataSet[game_id][1][playerId].append(dataSet[game_id][1][playerId][-1].copy())
            for playerId in dataSet[game_id][2]:
                dataSet[game_id][2][playerId].append(dataSet[game_id][2][playerId][-1].copy())

        elif state['id'] in dataSet[game_id][2]:
            f_new = get_features_from_state(state, dataSet[game_id][2][state['id']][-1])
            if f_new == dataSet[game_id][2][state['id']][-1]:
                # The states are same as previous state.
                return
            for playerId in dataSet[game_id][2]:
                if state['id'] == playerId:
                    dataSet[game_id][2][playerId].append(f_new)
                else:
                    dataSet[game_id][2][playerId].append(dataSet[game_id][2][playerId][-1].copy())
            for playerId in dataSet[game_id][1]:
                dataSet[game_id][1][playerId].append(dataSet[game_id][1][playerId][-1].copy())


truncated_data_path = "C:/Users/syeda/PycharmProjects/PredictingVictories/raw data/tt_logs_short/"

fileNames = []
for (dirpath, dirnames, fNames) in walk(truncated_data_path):
    fileNames.extend(fNames)
    break

dataSet = {}

for fileName in fileNames:
    game_id = fileName.partition(".")[0]
    # Label - who won the game
    # Player 0's data
    # Player 1's data
    dataSet[game_id] = [-1, {}, {}]
    with open(truncated_data_path + fileName) as f:
        data = json.load(f)
        objects = data['RegisteredObjects']
        for obj in objects:
            if obj['classId'] is not None and obj['classId'].startswith("unt"):
                if obj["ownerId"] == 0:
                    dataSet[game_id][1][obj["id"]] = [[1.0, 1.0] + get_binary_encoding(unitTypeMap[obj['asset']])]
                else:
                    dataSet[game_id][2][obj["id"]] = [[1.0, 1.0] + get_binary_encoding(unitTypeMap[obj['asset']])]

        for state in data["States"]:
            if state['t'] == 0.0:
                continue

            update_dataset_with_new_state(dataSet, state)

with open("preprocessed_data.pkl", 'wb') as f:
    pickle.dump(dataSet, f)
