import json
import pickle
from os import walk

tabular_data_path = "C:/Users/syeda/PycharmProjects/PredictingVictories/raw data/tt_logs_flattened/"

weaponIds = ['wpn01', 'wpn02', 'wpn03', 'wpn05', 'wpn06', 'wpn08', 'wpn09', 'wpn10', 'wpn11', 'wpn12', 'wpn13', 'wpn14',
             'wpn15', 'wpn16', 'wpn17', 'wpn18', 'wpn19', 'wpn20', 'wpn21', 'wpn22', 'wpn23', 'wpn24', 'wpn25', 'wpn26',
             'wpn27', 'wpn28', 'wpn29', 'wpn30', 'wpn31', 'wpn32', 'wpn33', 'wpn34']

weaponIdsMap = {}
for i in range(len(weaponIds)):
    weaponIdsMap[weaponIds[i]] = i + 1


def get_binary_encoding(num):
    res = []
    for i in range(6):
        res.append(num % 2)
        num //= 2
    return list(reversed(res))


def get_weapon_encoding(weaponIds):
    res = []
    if len(weaponIds) > 0:
        res += get_binary_encoding(weaponIdsMap[weaponIds[0]])
    else:
        res += get_binary_encoding(0)
    if len(weaponIds) > 1:
        res += get_binary_encoding(weaponIdsMap[weaponIds[1]])
    else:
        res += get_binary_encoding(0)
    return res


dataSet = {}

with open('preprocessed_data.pkl', 'rb') as f:
    dataSet = pickle.load(f)

fileNames = []
for (dirpath, dirnames, fNames) in walk(tabular_data_path):
    fileNames.extend(fNames)
    break

for fileName in fileNames:
    game_id = fileName.partition(".")[0]
    with open(tabular_data_path + fileName) as f:
        data = json.load(f)
        game_state = data['State']
        for obj in game_state:
            if obj['id'] in dataSet[game_id][1]:
                dataSet[game_id][1][obj['id']].append(
                    [round(obj['State']['ap']['current'] / obj['State']['ap']['max'], 2),
                     round(obj['State']['hp']['current'] / obj['State']['hp']['max'], 2)] +
                    dataSet[game_id][1][obj['id']][-1][2:])
                for timeStep in dataSet[game_id][1][obj['id']]:
                    if 'eq' in obj and 'weaponIds' in obj['eq']:
                        timeStep.extend(get_weapon_encoding(obj['eq']["weaponIds"]))
                    else:
                        timeStep.extend(get_weapon_encoding([]))
            elif obj['id'] in dataSet[game_id][2]:
                dataSet[game_id][2][obj['id']].append(
                    [round(obj['State']['ap']['current'] / obj['State']['ap']['max'], 2),
                     round(obj['State']['hp']['current'] / obj['State']['hp']['max'], 2)] +
                    dataSet[game_id][2][obj['id']][-1][2:])
                for timeStep in dataSet[game_id][2][obj['id']]:
                    if 'eq' in obj and 'weaponIds' in obj['eq']:
                        timeStep.extend(get_weapon_encoding(obj['eq']["weaponIds"]))
                    else:
                        timeStep.extend(get_weapon_encoding([]))

with open("flattened_preprocessed_data.pkl", 'wb') as f:
    pickle.dump(dataSet, f)
