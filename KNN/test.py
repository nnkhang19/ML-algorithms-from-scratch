import json
import matplotlib.pyplot as plt

with open("norm1.json",'r') as file:
    acc = json.load(file)


values = list(acc.values())[1:]
keys = list(acc.keys())[1:]

number_of_neighbors = list(map(int, keys))


plt.plot(number_of_neighbors, values)
plt.ylabel("Accuracy")
plt.show()

max_acc = max(values)

neighbor_with_max_acc = keys[values.index(max_acc)]

print(max_acc)
print(neighbor_with_max_acc)

