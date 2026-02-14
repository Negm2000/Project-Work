import json
import base64
import os

nb_path = r'e:\Project-Work\Report\Final Code\pcl_presence_training_AddedCells.ipynb'
output_dir = r'e:\Project-Work\Report\Final Code\report\figures'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if 'outputs' in cell:
        for j, output in enumerate(cell['outputs']):
            if 'data' in output and 'image/png' in output['data']:
                data = output['data']['image/png']
                # Check for multiple lines in base64 (common in ipynb)
                if isinstance(data, list):
                    data = "".join(data)
                filename = f"cell_{i}_out_{j}.png"
                with open(os.path.join(output_dir, filename), "wb") as fh:
                    fh.write(base64.b64decode(data))
                print(f"Saved {filename}")
