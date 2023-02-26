import streamlit as st

import torch
from yolov5 import detect
from ultralytics import YOLO

import cv2




import os
import shutil


from PIL import Image
def load_image(image_file):
    print("Loading ",image_file)
    img = Image.open(image_file)
    img.save(os.path.join("data",image_file.name))
    return img



def app():
    cur_dir=os.getcwd()
    header=st.container()
    result_all = st.container()
    with header:
        st.subheader("Detect crop and weed in image")
        image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            st.write(file_details)

            # To View Uploaded Image
            st.image(load_image(image_file)
                ,width=250
                )
            fname=image_file.name
            image_file=os.path.join("data",fname)
        else:
            fname="test_img.jpg"
            proxy_img_file="data/"+fname
            img = Image.open(proxy_img_file)
            image_file=os.path.join(cur_dir,proxy_img_file)
            st.image(img,width=250)            


    with result_all:     
        weight_path=os.path.join(cur_dir,"yolov5/runs/train/TrainModel/weights/best.pt")
        shutil.rmtree('yolov5/runs/detect/')

        detect.run(weights=weight_path,name="TestModel", source=image_file)  
        image_file_output="yolov5/runs/detect/TestModel/"+fname
        img = Image.open(image_file_output)
        st.subheader("Crop and Weed Detections")    
        st.image(img,width=250)            
        
        # for yolo8 versions
        model_y8n = YOLO("yolov8/models/nano/best.pt") # pass any model type
        results = model_y8n.predict(source=image_file, save=False, save_txt=False)  # save predictions as labels
        res_plotted = results[0].plot()        
        st.subheader("Yolov8 Nano Model")
        st.image(res_plotted,width=250)            

        model_y8s = YOLO("yolov8/models/small/best.pt") # pass any model type
        results = model_y8s.predict(source=image_file, save=False, save_txt=False)  # save predictions as labels
        res_plotted = results[0].plot()        
        st.subheader("Yolov8 Small Model")
        st.image(res_plotted,width=250)    


        model_y8m = YOLO("yolov8/models/medium/best.pt") # pass any model type
        results = model_y8m.predict(source=image_file, save=False, save_txt=False)  # save predictions as labels
        res_plotted = results[0].plot()        
        st.subheader("Yolov8 Medium Model")
        st.image(res_plotted,width=250)    


        model_y8l = YOLO("yolov8/models/large/best.pt") # pass any model type
        results = model_y8l.predict(source=image_file, save=False, save_txt=False)  # save predictions as labels
        res_plotted = results[0].plot()        
        st.subheader("Yolov8 Large Model")
        st.image(res_plotted,width=250)            

        
        







        