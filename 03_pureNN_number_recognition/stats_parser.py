import json

from matplotlib import pyplot as plt

file = open("results_json/best_res", "r")
dict_res = json.load(file)
list_LRs = []
list_prfmnc = []
for dictt in dict_res:
    list_LRs.append(dictt["hidden_nodes"])
    list_prfmnc.append(dictt["performance"])
# valuesx = range(len(list_LRs))
# plt.xticks(valuesx, list_LRs)
plt.plot(list_LRs,list_prfmnc, color='black', marker='o', markerfacecolor='red')

plt.xlabel('Number of hidden nodes')
plt.ylabel('Performance')
plt.show()
