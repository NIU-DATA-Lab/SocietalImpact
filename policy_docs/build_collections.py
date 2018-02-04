import os
import json
import pickle
import progressbar
data = "/home/christian/data/data"

doi_index = {}
policy_collection = set()
non_policy_collection = set()
error_log = ""

bar = progressbar.ProgressBar()

for root, subFolders, files in os.walk(data):
    print(root)
    for file in files:
        if file[-5:] == ".json":
            try:
                with open(os.path.join(root, file), 'r') as f:
                    f = json.load(f)
                    try:
                        doi_index[f["citation"]["doi"]] = os.path.join(root, file)
                    except KeyError:
                        error_log += "KeyError in {} when finding doi.\n".format(os.path.join(root, file))
                    if f.get("citation").get("abstract", False):
                        if f.get("posts", False).get("policy", False):
                            policy_collection.add(os.path.join(root, file))
                        else:
                            non_policy_collection.add(os.path.join(root, file))
            except:
                error_log += "some error in {}\n".format(os.path.join(root, file))
                pass

with open(os.path.join(data, "doi.pickle"), "wb") as f:
    pickle.dump(doi_index, f)

with open(os.path.join(data, "policy.pickle"), "wb") as f:
    pickle.dump(policy_collection, f)

with open(os.path.join(data, "non_policy.pickle"), "wb") as f:
    pickle.dump(non_policy_collection, f)

print(error_log)

