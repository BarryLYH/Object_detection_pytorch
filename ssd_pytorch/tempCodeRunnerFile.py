for img in data:
    line = str(img)
    line = "0" * (12 - len(img)) + line + " "
    w, h  = data[img]["wh"]
    for bb in data[img]["bbox"]:
        cx, cy, bw, bh, cls = bb
        xmin = round((cx - bw/2) / w, 4)
        xmax = round((cx + bw/2) / w, 4)
        ymin = round((cy - bh/2) / h, 4)
        ymax = round((cy + bh/2) / h, 4)
        s = str(xmin) + "," + str(ymin) + "," + \
            str(xmax) + "," + str(ymax) + "," + \
            str(cls) + " "
        line = line + s
    line = line[:-1]
    print(line)
