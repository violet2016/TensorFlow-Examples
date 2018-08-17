import json
import os
cluster = {'ps': ['localhost:2224'],
           'worker': ['localhost:2226', 'localhost:2227']}
os.environ['TF_CONFIG'] = json.dumps(
    {'cluster': cluster,
     'task': {'type': 'ps', 'index': 0}})
