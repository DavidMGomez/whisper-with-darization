"""Read deployment metadata only: never fetch predictions, inputs or hearing contents."""
import json
import os
import urllib.request


def get(path):
    req = urllib.request.Request('https://api.replicate.com/v1/' + path,
        headers={'Authorization': 'Bearer ' + os.environ['REPLICATE_API_TOKEN']})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


deployment = get('deployments/xiriustech/xirius-whisper-fast')
print('deployment_version', deployment.get('current_release', {}).get('version'))
model = get('models/xiriustech/whisper-with-darization')
print('latest_published_version', (model.get('latest_version') or {}).get('id'))
