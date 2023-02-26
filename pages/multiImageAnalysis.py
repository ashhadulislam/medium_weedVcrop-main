import streamlit as st

import torch
from yolov5 import detect
from ultralytics import YOLO

import cv2

import zipfile

import shutil

import os
import shutil
import time
 

from PIL import Image
def load_image(image_file):
    print("Loading ",image_file)
    img = Image.open(image_file)
    img.save(os.path.join("data",image_file.name))
    return img



def app():
    cur_dir=os.getcwd()
    header=st.container()
    selection=st.container()
    result_all = st.container()

    options={
        "Nano":"nano",
        "Small":"small",
        "Medium":"medium",
        "Large":"large",
        
        }

    all_yolov8_models={}
    for k,v in options.items():
        all_yolov8_models[v]=YOLO("yolov8/models/"+v+"/best.pt")

    with header:
        st.subheader("Detect crop and weed in multiple image")
        zip_file = st.file_uploader("Upload Zip file containing multiple images", type=["zip"])

        for dir in os.listdir(os.path.join("data","unzipped")):
            if ".DS_Store" in dir:
                continue

            if os.path.isdir(os.path.join("data","unzipped",dir)):
                shutil.rmtree(os.path.join("data","unzipped",dir))        
            else:
                os.remove(os.path.join("data","unzipped",dir))


        if zip_file is not None:

            ts = str(time.time())            
            unzip_location=os.path.join("data","unzipped",ts)            
            if not os.path.isdir(unzip_location):
                os.mkdir(unzip_location)                        

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_location)              


        else:


            proxy_zip_file="data/Archive.zip"            
            

            ts = str(time.time())
            
            
                
            unzip_location=os.path.join("data","unzipped",ts)            
            if not os.path.isdir(unzip_location):
                os.mkdir(unzip_location)            


            with zipfile.ZipFile(proxy_zip_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_location)  



      


    with selection:
        st.header("Select the model version you would like to run")
        

        model_selection_checkboxes=[]
        count=0
        for k,v in options.items():
            if count==0:
                value=True
            else:
                value=False
            model_selection_checkboxes.append(st.checkbox(k,value=value),)
            count+=1            
            

    with result_all:     

        # let us create a fodler to store the results
        
        for dir in os.listdir(os.path.join("data","results")):
            if ".DS_Store" in dir:
                continue
            if os.path.isdir(os.path.join("data","results",dir)):
                shutil.rmtree(os.path.join("data","results",dir))        
            else:
                os.remove(os.path.join("data","results",dir))
        results_location="data/results/"+ts
        if not os.path.isdir(results_location):
            os.mkdir(results_location)
        
        all_model_names=list(options.values())
        for i in range(len(model_selection_checkboxes)):
            if model_selection_checkboxes[i]==True:
                chosen_model_name=all_model_names[i]
                chosen_model=all_yolov8_models[chosen_model_name]
                results_location_model=os.path.join(results_location,chosen_model_name)

                if not os.path.isdir(results_location_model):
                    os.mkdir(results_location_model)


                results = chosen_model.predict(source=unzip_location, save=False,save_txt=False)
                counter=0
                for r in results:
                    res_plotted = r[0].plot()        
                    # print("Plotted image = ",res_plotted.shape)
                    cv2.imwrite(results_location_model+"/"+str(counter)+".jpeg",res_plotted)
                    counter+=1


        # can we compress the files here
        output_zip_filename=results_location
        shutil.make_archive(output_zip_filename, 'zip', results_location)


        with open(output_zip_filename+".zip", "rb") as fp:
            btn = st.download_button(
                label="Download ZIP",
                data=fp,
                file_name="Results.zip",
                mime="application/zip"
            )



        