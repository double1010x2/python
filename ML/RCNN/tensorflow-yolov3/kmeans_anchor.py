import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "/Users/vincentwu/Documents/GitHub/1st-DL-CVMarathon/homework/finalProject/data/xml_all"
#ANNOTATIONS_PATH = "annotations/annotations.csv"
CLUSTERS = 9


def load_dataset(path):
  dataset = []
  data_size = []
  for xml_file in glob.glob("{}/*xml".format(path)):
    tree = ET.parse(xml_file)

    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))
    data_size.append([height, width])
    for obj in tree.iter("object"):
      xmin = int(obj.findtext("bndbox/xmin")) / width
      ymin = int(obj.findtext("bndbox/ymin")) / height
      xmax = int(obj.findtext("bndbox/xmax")) / width
      ymax = int(obj.findtext("bndbox/ymax")) / height

      xmin = np.float64(xmin)
      ymin = np.float64(ymin)
      xmax = np.float64(xmax)
      ymax = np.float64(ymax)
      if xmax == xmin or ymax == ymin:
         print(xml_file)
      dataset.append([xmax - xmin, ymax - ymin])
  return np.array(dataset), np.array(data_size)

if __name__ == '__main__':
  #print(__file__)
  import pdb; pdb.set_trace()
  #data = pd.read_csv(ANNOTATIONS_PATH)
  #height = (data.height.values).astype(np.int)
  #width  = (data.width.values).astype(np.int)
  #xmin   = (data.xmin.values.astype(np.int)/width).astype(np.float64)
  #xmax   = (data.xmax.values.astype(np.int)/width).astype(np.float64)
  #ymin   = (data.ymin.values.astype(np.int)/height).astype(np.float64)
  #ymax   = (data.ymax.values.astype(np.int)/height).astype(np.float64)
  ##data = data[["xmin", "xmax", "ymin", "ymax"]].values
  #data = np.c_[xmax-xmin, ymax-ymin]

  data, data_size = load_dataset(ANNOTATIONS_PATH)
  out = kmeans(data, k=CLUSTERS)
  #clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
  #out= np.array(clusters)/416.0
  print(out)
  print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
  print("Boxes:\n {}-{}".format(out[:, 0]*416, out[:, 1]*416))

  ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
  print("Ratios:\n {}".format(sorted(ratios)))
