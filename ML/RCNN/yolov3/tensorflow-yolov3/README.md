Yolov3 by tensorflow for animal recognition on Raccoon and Kangaroo
==========
# Abstract
> Use the tensorflow yolov3 by YunYang [github](https://github.com/YunYang1994/tensorflow-yolov3)
 to recognize the kangaroo and raccoon. Encountered some difficulty about syntax change on different tensorflow version, poor prediction results by low confidence value but low loss value and some bug in this reference Tensorflow yolov3.  Some is fixed, but low confidence value is improved a little. Even the model performance is not good to recognize the picture, but I got a lot in this final project.  

## code function
-[`kmeans_anchor.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/kmeans_anchor.py)
clustering the anchor information for kangaroo and raccoon data 
```
    python kmeans_anchor.py
```
-[`xml_to_csv.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/xml_to_csv.py)
collect xml file to csv and txt file (anchor format: *.jpg xmin,ymin,xmax,ymax,class)
```
    python xml_to_anchor.py
```
-[`train.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/train.py)
training the yolov3 with darknet53 network 
```
    python train.py
```
-[`freeze_graph.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/freeze_graph.py)
transfer *ckpt file to *.pb file
```
    python freeze_graph.py *.ckpt* 
```
-[`image_demo.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/image_demo.py)
prediction yolov3 model on one image
```
    python image_demo.py *.pb *.jpg
```
-[`videl_demo.py`](https://github.com/double1010x2/1st-DL-CVMarathon/blob/master/homework/finalProject/tensorflow-yolov3/videl_demo.py)
prediction yolov3 model on one videl
```
    python videl_demo.py *.pb *.jpg
```
