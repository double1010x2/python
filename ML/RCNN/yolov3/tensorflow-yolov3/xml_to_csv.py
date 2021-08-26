import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.utils import shuffle
TRAIN_RATIO = 0.8
TEST_RATIO  = 1.-TRAIN_RATIO

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     1 if member[0].text == "kangaroo" else 0
                     )
            xml_list.append(value)
    #column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    column_name = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    image_path  = path.replace('annotations', 'images')
    xml_df.filename = image_path + xml_df.filename.values
    return xml_df

def random_choose(df, ratio):
    df_size = len(df)
    extract_size    = int(df_size*ratio)
    index_rand      = np.random.default_rng().choice(df_size, size=extract_size, replace=False)
    ff              = np.zeros(df_size, bool)
    ff[index_rand]= True
    df_extract      = df.iloc[ff]
    df_keep         = df.iloc[~ff]
    return df_extract, df_keep, index_rand

def write_txt(df, filename):
    df_mat = df.as_matrix()
    row, col = df_mat.shape
    f_write = open(filename, "w")
    for ri in range(row):
        line = "%s " %df_mat[ri][0]
        #f_write.write("%s " %df_mat[ri][0])
        for ci in range(col-1):
            #f_write("%s" %df_mat[ri][ci+1])
            line += str(df_mat[ri][ci+1])
            if ci != col-2:
                #f_write(",")
                line += ","
        if ri != row-1:
            #f_write("\n")
            line += "\n"
        f_write.write(line)
    f_write.close()

def main():
    image_path1= "/Users/vincentwu/Documents/GitHub/1st-DL-CVMarathon/homework/finalProject/data/kangaroo/annotations/" #os.path.join(os.getcwd(), 'annotations')
    image_path2= "/Users/vincentwu/Documents/GitHub/1st-DL-CVMarathon/homework/finalProject/data/raccoon_dataset/annotations/" #os.path.join(os.getcwd(), 'annotations')
    xml_df1 = xml_to_csv(image_path1)
    xml_df2 = xml_to_csv(image_path2)
    df1_train, df1_test, df1_index = random_choose(xml_df1, TRAIN_RATIO)
    df2_train, df2_test, df2_index = random_choose(xml_df2, TRAIN_RATIO)
    df_train = df1_train.append(df2_train)
    df_test  = df1_test.append(df2_test)
    df_train = shuffle(df_train)
    df_test  = shuffle(df_test)
    write_txt(df_train, 'train.txt')
    write_txt(df_test, 'test.txt')
    df_train.to_csv('train.csv', index=None)
    df_test.to_csv('test.csv', index=None)
    print('Successfully converted xml to csv and txt.')


main()
