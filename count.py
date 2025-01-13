import os
import re

count = 0
names = []
# go through all files in "dynamic_GLM_figures"
for root, dirs, files in os.walk("dynamic_GLM_figures"):
    for file in files:
        # if the file starts with "meta_state_development_" and ends with "_fitvar_0.04.png" and contain the string "True_first", count it
        if file.startswith("meta_state_development_") and file.endswith("_fitvar_0.04.png") and "True_first" in file:
            count += 1
            # using regular expression, extract the mouse name between "meta_state_development_" and "_fitvar_0.04.png"
            mouse_name = re.search(r"meta_state_development_(.*?)_True_first.*", file).group(1)
            names.append(mouse_name)

print(count)
print(names)

cluster_names = []
# read the text file cluster_names.txt line by line
with open("cluster_names.txt", "r") as f:
    lines = f.readlines()
    # go through each line
    for line in lines:
        # using regular expression, extract the name between "canonical_result_" and "prebias_var_0.04.p"
        cluster_name = re.search(r"canonical_result_(.*?)_prebias_var_0.04.p", line).group(1)
        cluster_names.append(cluster_name)

print(len(cluster_names))

print('missing')
# print all names in cluster_names that are not in names
for cluster_name in cluster_names:
    if cluster_name not in names:
        print(cluster_name)