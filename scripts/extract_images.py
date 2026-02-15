import json
import base64
import os

nb_path = r'e:\Project-Work\Report\Final Code\pcl_presence_training_AddedCells.ipynb'
output_dir = r'e:\Project-Work\Report\Final Code\report\figures'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Mapping known cells to desired filenames
# This is a guess based on the cell list, I will extract several and see
targets = {
    6: "mask_definitions.png",
    34: "synthetic_samples.png", # Typical location for KO generation samples
    68: "gradcam_analysis.png",
    82: "feature_maps.png"
}

for i, cell in enumerate(nb['cells']):
    if i in targets or ('source' in cell and any('threshold_calculation' in s for s in cell['source'])):
        filename = targets.get(i, f"image_cell_{i}.png")
        if 'outputs' in cell:
            for j, output in enumerate(cell['outputs']):
                if 'data' in output and 'image/png' in output['data']:
                    data = output['data']['image/png']
                    with open(os.path.join(output_dir, filename), "wb") as fh:
                        fh.write(base64.b64decode(data))
                    print(f"Saved {filename}")
                    break
