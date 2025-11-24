import os
import glob
import pandas as pd
from PIL import Image
from xlsxwriter import Workbook

def coco_to_voc(c, x, y, w, h, img_width, img_height):
    xmin = (x - w / 2) * img_width
    ymin = (y - h / 2) * img_height
    xmax = (x + w / 2) * img_width
    ymax = (y + h / 2) * img_height
    return c, xmin, ymin, xmax, ymax

def process_txt_files(txt_files, img_folder):
    data = []
    for txt_file in txt_files:
        image_name = os.path.splitext(os.path.basename(txt_file))[0] + '.jpg'
        img_path = os.path.join(img_folder, image_name)
        img = Image.open(img_path)
        img_width, img_height = img.size

        if os.stat(txt_file).st_size == 0:
            data.append([image_name, None, None, None, None, None, None])
        else:
            with open(txt_file, 'r') as f:
                for line in f:
                    c, x, y, w, h, confidence = map(float, line.strip().split())
                    c, xmin, ymin, xmax, ymax = coco_to_voc(c, x, y, w, h, img_width, img_height)
                    data.append([image_name, int(c), xmin, ymin, xmax, ymax, confidence])
    return data



def main():
    columns = ['image_name', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence']
    output_file = 'output.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    target_folders = ['T1_0.05', 'T2_0.1', 'T3_0.15','T5_0.25','T6_0.35','T9_0.45']
    img_folder = 'images'
    for folder in target_folders:
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        data = process_txt_files(txt_files, img_folder)
        df = pd.DataFrame(data, columns=columns)
        # 将class列转换为整型
        df['class'] = df['class'].astype(int)
        df.loc[df['image_name'].str.contains('1'), 'class'] = '腺瘤性息肉'
        df.loc[df['image_name'].str.contains('0'), 'class'] = '增生性息肉'
        # 将xmin, ymin, xmax, ymax列转换为浮点型
        df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(float)
        df.to_excel(writer, sheet_name=folder, index=False)
        print(f'文件夹{folder}已完成')

    #writer.save()
    writer.close()

if __name__ == '__main__':
    main()