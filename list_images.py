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
                # Get the first line of source to help identify
                sources = cell.get('source', [])
                id_text = sources[0].strip()[:50] if sources else f"cell_{i}"
                print(f"Cell {i}, Output {j}: {id_text}")
