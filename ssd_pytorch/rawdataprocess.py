import json
import pandas as pd

#read coco val.json. Train data is too large, here i just want to show how ssd works

with open("/Users/Barry/Desktop/annotations/instances_val2017.json", "r") as f:
    temp = json.loads(f.read())
data = {}
label = {}
for lab in temp["categories"]:
    label[lab["id"]] = lab["name"]
print(label)
for img in temp["images"]:
    if img["id"] not in data:
        data[img['id']] = {"wh": [img["width"], img["height"]]}
class_num = []
for item in temp["annotations"]:
    idd = item["image_id"]
    bb = item["bbox"]
    ca = item["category_id"]

    if ca not in class_num: class_num.append(ca)
    bb.append(ca)
    if "bbox" in data[idd]:
        data[idd]["bbox"].append(bb)
    else:
        data[idd]["bbox"] = [bb]
class_num = sorted(class_num)
print(class_num)
# the class number in json is 90, but only 80 of them are useful
i = 1
with open("labels.txt", "w") as labput:
    for cl in class_num:
        labput.write(str(i) + " " + label[cl] + "\n")
        i += 1
cls_map = {}
for i in range(len(class_num)):
    cls_map[class_num[i]] = i + 1
print(cls_map)
with open("dataset.txt", "w") as output:
    for img in data:
        line = str(img)
        line = "/Users/Barry/Desktop/val2017/" + "0" * (12 - len(line)) + line + ".jpg "
        w, h  = data[img]["wh"]
        if "bbox" in data[img]:
            for bb in data[img]["bbox"]:
                xmin, ymin, bw, bh, cls = bb
                cls = cls_map[cls]
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

