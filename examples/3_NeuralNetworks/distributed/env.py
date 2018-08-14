import json
import os
cluster = {'ps': ['vm-instance-vc-ml-workload-worker-1:2224', 'vm-instance-vc-ml-workload-master:2224'],
           'worker': ['vm-instance-vc-ml-workload-worker-1:2225', 'vm-instance-vc-ml-workload-master:2225']}
os.environ['TF_CONFIG'] = json.dumps(
    {'cluster': cluster,
     'task': {'type': 'ps', 'index': 0}})
