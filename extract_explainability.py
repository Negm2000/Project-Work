import json
import base64
import os

nb_path = r'e:\Project-Work\Report\Final Code\pcl_presence_training_AddedCells.ipynb'
output_dir = r'e:\Project-Work\Report\Final Code\report\figures'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'Single Image Analysis' in source and 'Grad-CAM' in source:
        print(f"Found Single Image Analysis at cell {i}")
        if 'outputs' in cell:
            for j, output in enumerate(cell['outputs']):
                if 'data' in output and 'image/png' in output['data']:
                    data = output['data']['image/png']
                    if isinstance(data, list): data = "".join(data)
                    with open(os.path.join(output_dir, "gradcam_analysis.png"), "wb") as fh:
                        fh.write(base64.b64decode(data))
                    print(f"Saved gradcam_analysis.png from cell {i}")
    
    if 'Feature Map Visualization' in source:
        print(f"Found Feature Map Visualization at cell {i}")
        if 'outputs' in cell:
            for j, output in enumerate(cell['outputs']):
                if 'data' in output and 'image/png' in output['data']:
                    data = output['data']['image/png']
                    if isinstance(data, list): data = "".join(data)
                    with open(os.path.join(output_dir, "feature_maps.png"), "wb") as fh:
                        fh.write(base64.b64decode(data))
                    print(f"Saved feature_maps.png from cell {i}")
