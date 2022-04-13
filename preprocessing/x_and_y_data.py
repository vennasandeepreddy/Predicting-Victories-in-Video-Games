import pickle

dataSet = {}
with open('training_data.pkl', 'rb') as f:
    dataSet = pickle.load(f)

x = []
y = []

for game in dataSet:
    # New matrix per game
    x.append([])
    time_steps = 0
    for val in dataSet[game][1].values():
        time_steps = len(val)
        break
    for i in range(time_steps):
        x[-1].append([])
        for player_id in dataSet[game][1]:
            x[-1][-1].extend(dataSet[game][1][player_id][i])
        for player_id in dataSet[game][2]:
            x[-1][-1].extend(dataSet[game][2][player_id][i])
    y.append(float(dataSet[game][0]))

del dataSet

with open('x.pkl', 'wb') as f:
    pickle.dump(x, f)

with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)
