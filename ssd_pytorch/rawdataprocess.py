import json
import pandas as pd

#read coco val.json. Train data is too large, here i just want to show how ssd works

with open("/Users/Barry/Desktop/annotations/instances_val2017.json", "r") as f:
    temp = json.loads(f.read())
data = {}
for img in temp["images"]:
    if img["id"] not in data:
        data[img['id']] = {"wh": [img["width"], img["height"]]}
for item in temp["annotations"]:
    idd = item["image_id"]
    bb = item["bbox"]
    ca = item["category_id"]
    bb.append(ca)
    if "bbox" in data[idd]:
        data[idd]["bbox"].append(bb)
    else:
        data[idd]["bbox"] = [bb]
with open("dataset.txt", "w") as output:
    for img in data:
        line = str(img)
        line = "0" * (12 - len(line)) + line + " "
        w, h  = data[img]["wh"]
        if "bbox" in data[img]:
            for bb in data[img]["bbox"]:
                xmin, ymin, bw, bh, cls = bb
                xmax = min(round((xmin + bw) / w, 4), 1)
                ymax = min(round((ymin + bh) / h, 4), 1)
                xmin = max(round(xmin / w, 4), 0)
                ymin = max(round(ymin / h, 4), 0)
                
                s = str(xmin) + "," + str(ymin) + "," + \
                    str(xmax) + "," + str(ymax) + "," + \
                    str(cls) + " "
                line = line + s
            line = line[:-1]
        output.write(line)
        output.write("\n")

