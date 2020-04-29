import os, csv
from PIL import Image

with open(
        "C:/Users/Administrator/PycharmProjects/MIT_YOLO/dataset/our_test.csv",
        'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(['Name', 'URL', 'Width', 'Height', 'Scale', 'X0, Y0, H0, W0', 'X1, Y1, H1, W1', 'etc'])
    for path, dirs, files in os.walk("C:/Users/Administrator/PycharmProjects/MIT_YOLO/Cones/Combo_img"):
        for filename in files:
            no_BB = 0
            full_path = path + '/' + filename
            new_path = full_path.replace(os.sep, '/')
            is_img = filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
            if is_img:
                print(new_path)
                im = Image.open(new_path)
                width, height = im.size
                print("w = ", width, "h = ", height)
                txt_path = new_path.replace('png', 'txt')
                txt_path = txt_path.replace('jpg', 'txt')
                print(txt_path)
                try:
                    file = open(txt_path, 'r')
                except:
                    no_BB = 1

                if no_BB == 1:  # no BB in image
                    row = [filename, 'N/A', width, height, 1.028169]
                    print('row = ', row)
                    writer.writerow(row)
                else:  # there is at least one BB
                    BB = []
                    for line in file:
                        # print('original line: ', line)
                        x = float(line[2:10]) * width
                        y = float(line[11:20]) * height
                        w = round(float(line[20:28]) * width)
                        h = round(float(line[29:40]) * height)
                        x0 = str(round(x - w/2))
                        y0 = str(round(y - h/2))
                        w = str(w)
                        h = str(h)
                        # print('x = ', x, 'y = ', y, 'w = ', w, 'h = ', h)
                        new_line = '[' + x0 + ',' + y0 + ',' + h + ',' + w + ']'
                        # print('new line: ', new_line)
                        BB.append(new_line)
                        # print('BB = ', BB)
                    row = [filename, 'N/A', width, height, 1.028169]
                    row = row + BB
                    print('row = ', row)
                    writer.writerow(row)
