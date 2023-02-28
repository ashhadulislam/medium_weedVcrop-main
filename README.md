# Welcome to Farm-Eye

 **Farm-Eye** is an AI based computer vision tool capable of identifying crops and weeds from images.

![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*HF_Bjk4gMPohESQggXaW7g.jpeg)


# 01 Motivation

**Intel® oneAPI Hackathon for Open Innovation** is a recent [competition](https://www.hackerearth.com/challenges/hackathon/intel-oneapi-hackathon-for-open-innovation/) hosted by intel where we need to detect the regions in pictures of croplands where weeds are present. The product developed will be helpful for the targeted application of pesticides to get rid of weeds. Weeds are unwanted trespassers in the agricultural business. They deny the nutrients, water, land, and other critical resources to grow healthy crops. Weed can lead to lower yields and inefficient deployment of resources by farmers. Although pesticides are used to remove weeds, indiscriminate usage of pesticides creates health risks for humans. Let’s leverage computer vision and deep learning to detect the presence of weeds. This will enable targeted remediation techniques to remove them from fields with minimal environmental impact.

# 02 Setup
As we have used intelOneApi devcloud, you need to have a free account in intel OneApi. Here is the [link](https://www.intel.com/content/www/us/en/my-intel/sign-in.html?redirect=https://www.intel.com/content/www/us/en/forms/idz/devcloud-registration.html?tgt=https://www.intel.com/content/www/us/en/secure/forms/devcloud-enrollment/account-provisioning.html?eventcode=oneapiapjdec09).

![](https://miro.medium.com/max/1400/1*bSJl-WLup5TLXI7EIhZ0dA.png)

Sign in/up to use the powerful intel devcloud

## ssh into devcloud

Once you have an account you need to follow some steps to connect to the oneapi devcloud server machine. Go to this  [link](https://devcloud.intel.com/oneapi/get_started/aiAnalyticsToolkitSamples/)  to and follow the necessary instructions.

![](https://miro.medium.com/max/1400/1*fUYXnV1MVM8p-3khG-LCuQ.png)

Choose your operating system

You can choose option1: Automated Configuration which gives you a .txt file. You can run this file with the bash command and that will set up the configurations required to connect to devcloud.

![](https://miro.medium.com/max/1400/1*aEZjYRij3ozelD4Wrlmpgw.png)

Automated Configuration

Once the configuration is complete, just type ssh devcloud to connect to the server. You might have to type “yes” for the first time and re connect.

![](https://miro.medium.com/max/616/1*aCkovalBmrAbpJQnAyGyDQ.png)

This is what my terminal looks like after connecting to devcloud.

![](https://miro.medium.com/max/1400/1*HhGcBqVtChx0sKM_gHdPgg.png)

# 03 Project Setup
Create a folder called project at the home location with the following command

```
mkdir project  
cd project
```
## Data
Download the data from  [here](https://s3-ap-southeast-1.amazonaws.com/he-public-data/Weed_Detection5a431d7.zip). It is a collection of images of crops and weeds. Crop is labeleld 0 while weed is labelled 1.

Since you are on the ssh terminal, you can run a wget command to download the data. Make sure that you are inside the project folder.

```
wget https://s3-ap-southeast-1.amazonaws.com/he-public-data/Weed_Detection5a431d7.zip  
unzip Weed_Detection5a431d7.zip  
rm Weed_Detection5a431d7.zip  
mkdir Weed_Detection5a431d7  
mv classes.txt Weed_Detection5a431d7/  
mv data/ Weed_Detection5a431d7/
```

The additional commands arrange your files into another folder called Weed_Detection5a431d7. Now your files look as follows.

![](https://miro.medium.com/max/1400/1*rifkOKEezEs6Pv32S-JC7g.png)

Weed_Detection5a431d7 is the folder that you download from  [here](https://s3-ap-southeast-1.amazonaws.com/he-public-data/Weed_Detection5a431d7.zip).

The data folder contains the images and their labels. If you look at the agri_0_3.txt file, you will notice this:

![](https://miro.medium.com/max/1132/1*6hIH58_UvFDolppY6Sqq-A.png)

First value is label, followed by x and y coordinates of the center of the identified object and the length and width of the same

The above implies that the picture agri_0_3.jpeg contains one instance of crop (0).

![](https://miro.medium.com/max/1024/1*ILBPLQtKugaTz8UHw7Iyag.png)

agri_0_3.jpeg

![](https://miro.medium.com/max/524/1*oITZyILvFsSHc22bu4ynmQ.png)

Applying the co ordinates in the file, we see that the crop is getting highlighted

On the other hand, the file agri_0_6.txt looks as follows.

![](https://miro.medium.com/max/1096/1*_wJpFLjbOHWZGiwWXHmNzg.png)

First value is label, followed by x and y coordinates of the center of the identified object and the length and width of the same

The above implies that the image contains one instance of weed at the mentioned location.

![](https://miro.medium.com/max/1024/1*ZZgVY0Q43-COAJuDlu10Kw.png)

agri_0_6.jpeg

Using the co ordinates, the weed is highlighted as follows

![](https://miro.medium.com/max/524/1*PzZcvC3h_LqiiZT6t0UvMA.png)

The ambition is to train a model that is capable of separating the weed from the crops. Following tools will be used

**Framework**: Pytorch
**Model**: Yolov5/8

## Data Organization
As we will use yolo as our base model, we need to arrange our files so that the model can process them. This means having a separate folder for images and another one for annotations. These two folders should again contain three folders: train, val, and test. Thus the final folder structure should be as follows.

![](https://miro.medium.com/max/1400/1*IgPS_AFINQbu-YqxAJOkpQ.png)

The highlighted part needs to be generated

We shall take the files in the data folder and distribute them into images and annotations. The image files in the train folder should be matched by the annotations .txt file in the train folder and so on. Let us write a small code to ensure matching files go into the different folders. Let us create a python file called 01_DistributeImages.py.

![](https://miro.medium.com/max/1116/1*sMZIoSe_G9GiH9CUtVSx3Q.png)

Create the 01_DistributeImages.py file in the root folder of the project

Open the file using vi command and paste the below code.
```
vi 01_DistributeImages.py
```
```
import os  
from sklearn.model_selection import train_test_split  
import shutil  
  
# get all the images and annotation files in two lists  
images=[]  
annotations=[]  
for i in os.listdir("Weed_Detection5a431d7/data/"):  
    if ".txt" in i:  
        annotations.append(os.path.join("Weed_Detection5a431d7/data/",i))  
    elif ".DS_Store" in i:  
        pass  
    else:  
        images.append(os.path.join("Weed_Detection5a431d7/data/",i))  
          
# sort the files so that the order of images and annotations are same   
images.sort()  
annotations.sort()  
  
print("Number of images",len(images),"\nNumber of annotation files",len(annotations))  
  
  
  
# Split the dataset into train-valid-test splits   
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations,   
                                                                                test_size = 0.2, random_state = 1)  
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,   
                                                                              test_size = 0.5, random_state = 1)  
  
print("Training:",len(train_images),";Validation: ",len(val_images),";Test:",len(test_images))  
  
# create additional folders  
  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data")  
      
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/images"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/images")  
  
      
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/labels"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/labels")  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/images/train"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/images/train")  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/images/val"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/images/val")  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/images/test"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/images/test")  
      
      
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/labels/train"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/labels/train")  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/labels/val"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/labels/val")  
  
if not os.path.isdir("Weed_Detection5a431d7/assorted_data/labels/test"):  
    os.mkdir("Weed_Detection5a431d7/assorted_data/labels/test")          
  
      
#Utility function to move images   
def move_files_to_folder(list_of_files, destination_folder):  
    for f in list_of_files:  
#         print(f,destination_folder)  
        try:  
            shutil.copy(f, destination_folder)  
        except e:  
            print(e)  
            pass  
            # print(f,"Already there")  
            # assert False  
  
# Move the splits into their folders  

move_files_to_folder(train_images, 'Weed_Detection5a431d7/assorted_data/images/train/')  
move_files_to_folder(val_images, 'Weed_Detection5a431d7/assorted_data/images/val/')  
move_files_to_folder(test_images, 'Weed_Detection5a431d7/assorted_data/images/test/')  
move_files_to_folder(train_annotations, 'Weed_Detection5a431d7/assorted_data/labels/train/')  
move_files_to_folder(val_annotations, 'Weed_Detection5a431d7/assorted_data/labels/val/')  
move_files_to_folder(test_annotations, 'Weed_Detection5a431d7/assorted_data/labels/test/') 
```

Run the code as follows
```
python 01_DistributeImages.py 
```
![](https://miro.medium.com/max/1400/1*5N5eZq0L3jmgeIPo5nzEKg.png)

After moving the files

The first part of the code splits the file names into three lists (train, val, test), each for images and annotations. We ensure that the same files are present for images and annotations in each group. Then we create the test, train, and val folders. Finally, we copy the images from the data folder into the assorted_data folder.

Let us now check if the number of files in train, validation and test are consistent.
```
vi 01a_CountImages.py
```
```
import os  
  
print("Number of train images = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/images/train/")))  
print("Number of train annotations = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/labels/train/")))  
  
print("Number of val images = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/images/val/")))  
print("Number of val annotations = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/labels/val/")))  
  
print("Number of test images = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/images/test/")))  
print("Number of test annotations = ",len(os.listdir("Weed_Detection5a431d7/assorted_data/labels/test/")))

python 01a_CountImages.py
```
![](https://miro.medium.com/max/1292/1*HZCmoKzQBnuPtb9P0tSsyQ.png)

Number of files in the different folders

## End of data pre-processing

Believe me, the hard part is over. Now it is just about loading the yolov5 and training it on this data. If you have come this far, the rest is easy.

# 03 Training

## Get yolov5

We download the code repository of yolov5 from github in order to take advantage of the already existing codebase. Some commands need to be run from the terminal to download yolov5.
```
qsub -i  
source /opt/intel/inteloneapi/setvars.sh  > /dev/null 2>&1  
source activate pytorch  
cd project  
git clone https://github.com/ultralytics/yolov5  
pip install --user -r yolov5/requirements.txt
```
We first start the interactive mode to make use of the GPU. Then we activate the pytorch virtual environment. Next we download the github repository of yolov5 and install the necessary libraries. Note that we can only install different packages from the requirements.txt file if we are in the interactive mode.

![](https://miro.medium.com/max/1180/1*deEvPsl4aWhLzI1od6Vu1w.png)

A new folder called yolov5 will be created

## The YAML file

We need to create a YAML file that contains the particulars of training. The training, validation, and test folders must be mentioned in this file. The final YAML file looks like this, although we will create it programmatically.
```
train: /home/u178709/CropVWeedProject/Weed_Detection5a431d7/assorted_data/images/train  
val: /home/u178709/CropVWeedProject/Weed_Detection5a431d7/assorted_data/images/val  
test: /home/u178709/CropVWeedProject/Weed_Detection5a431d7/assorted_data/images/test  
nc: 2  
names: ['crop',  
        'weed',         
]
```
Above shows the YAML file containing particulars of train, test, validation, number of classes and names of classes

It is necessary to give the absolute path to the train, validation, and test folder. This is an important note. In order to reduce the confusion, we will develop the file using code. Create another python file 02CreateYAML.py
```
vi 02CreateYAML.py
```
Paste the following code snippet into the python file
```
import os  
from pathlib import Path  
path = Path(os.getcwd())  
final_data_path=os.path.join(path,"Weed_Detection5a431d7","assorted_data")  
print(final_data_path)  
  
yaml_data="train: "+final_data_path+"/images/train\n"  
yaml_data+="val: "+final_data_path+"/images/val\n"  
yaml_data+="test: "+final_data_path+"/images/test\n"  
  
rest_of_yaml_data='''nc: 2  
names: ['crop',  
        'weed',         
]'''  
  
yaml_data=yaml_data+rest_of_yaml_data  
print(yaml_data)  
  
yaml_file = open("dataDiff.yaml", "w")  
n = yaml_file.write(yaml_data)  
yaml_file.close()
```
The first part of the code gets the path to the training, validation and test files. The code results in creating a file called dataDiff.YAML at the root of the project.

![](https://miro.medium.com/max/1084/1*pNmHpgo8sdsW3vlzDDcKdw.png)

Creating the metadata for training

Next, we will run the code to perform training. The code to train is already written in yolov5/train.py. What we need to do is execute the code with the correct parameters. In Intel devcloud we need to do this by submitting a job to the system. That is done by creating a bash script containing the python command to train the model. This bash script is then added to the queue for execution.

Create a .sh file to contain the python command.
```
vi train_yolov5.sh
```
Paste the following lines into the .sh file
```
source /opt/intel/inteloneapi/setvars.sh  > /dev/null 2>&1  
source activate pytorch  
python yolov5/train.py --data dataDiff.yaml --cfg yolov5n.yaml --batch-size 32 --epochs 5 --name TrainModel
```
The first two lines are for the environment. An important parameter in the third line is —  _data_  where we give the name of the YAML file just created. Another necessary parameter is —  _epochs_  which are just 1 in the above example, but it should be changed to 20 or more to get better accuracy. Finally the parameter —  _name_  is important as it declares where the final trained model is going to be stored. For example, in this case, the trained model will be stored in the folder TrainModel  inside the run directory of the yolov5 folder. Also important is the cfg parameter, where we give the name of the model to be trained. The image below gives the different yolov5* versions. However we use a different one,  **yolov5n**  which is even smaller than yolov5s.

![](https://miro.medium.com/max/1400/1*bOc4rZ_gfO17PzF5l6hXGw.png)

Different sizes of YOLOv5 models

Now we can submit the training job. However you might face an error:
```
import pandas._libs.window.aggregations as window_aggregations   
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version   
`GLIBCXX_3.4.29' not found 
```
This is specific to the intel oneapi platform and can be resolved by adding the following line to the bash file running the training command. Modify the train_yolov5.sh file as follows:
```
source /opt/intel/inteloneapi/setvars.sh  
source activate pytorch  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/glob/development-tools/versions/oneapi/2023.0/oneapi/intelpython/latest/envs/pytorch/lib/  
python yolov5/train.py --data dataDiff.yaml --cfg yolov5n.yaml --batch-size 32 --epochs 5 --name TrainModel
```
The third line takes care of the library not found error.

We can now submit the job with the following command.
```
qsub -l nodes=1:gpu:ppn=2 -d . train_yolov5.sh 
```

Once the job is submitted, you can check the status with the below command.

```
watch -n 1 qstat -n -1
```

The progress can be seen as below:

![](https://miro.medium.com/max/1400/1*ZKHImLvyvpMxmpr0xcySkQ.png)

You can also find the training files inside the yolov5 folder. Go to yolov5/runs/train/TrainModel folder. You can see the intermediate efficacy of the model as and while it s being trained.

![](https://miro.medium.com/max/1400/1*0hiPTFh6asSVXYelRuZL_g.jpeg)

Model classification on validation images

You will know that your training is complete when qstat doesnt show any train_yolov5.sh any more. Also the yolov5/runs/train/TrainModel will have lot of files with validation results.

For example, below is the result on some validation images.

![](https://miro.medium.com/max/1378/1*ylEQ2Kto3-sS1JU2cC_HoQ.png)

Detecting weed and crop in validation images

![](https://miro.medium.com/max/1400/1*IFYwa2pCk3F1gNE7bVMPFQ.png)

Confusion matrix created after training

# 04 Test model

Once the model is trained satisfactorily, we want to test its performance. We create a folder called test_data and put some images in it for testing.

![](https://miro.medium.com/max/972/1*6X_aaj3qM7OiywCW1gPMJg.png)

Files and folders for testing

Notice how we have created a new file called test_yolov5.sh which will contain the commands to run the trained yolov5 model on the images present in the test_data folder. We transfer a couple of images into the test_data folder. Following is the content of the test_yolov5.sh file:
```
source /opt/intel/inteloneapi/setvars.sh  
source activate pytorch  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/glob/development-tools/versions/oneapi/2023.0/oneapi/intelpython/latest/envs/pytorch/lib/  
python yolov5/detect.py --source test_data/ --weights yolov5/runs/train/TrainModel/weights/best.pt --conf 0.25 --name TestModel
```
Note that this also has to be submitted as a job. The command for the same is as follows:
```
qsub -l nodes=1:gpu:ppn=2 -d . test_yolov5.sh
```
The output of the test will be stored in yolov5/runs/detect in a folder called TestModel.

![](https://miro.medium.com/max/500/1*fzHUMBcoUPt5o7rnmhJLdw.png)

# 05 Bonus — Streamlit App

Wouldn’t it be great if we could use this app in the web. We could just upload image of a crop and it gives the areas where weed is present. We can do exactly that using streamlit.

![](https://miro.medium.com/max/1200/1*bkMoiV4ErVFkZ355Ay9CfA.gif)

For a more detailed explanation of coding in streamlit, take a look at this  [article](https://pub.towardsai.net/deep-learning-a692669f6f42), and go to the section  _Hosting As a Streamlit Application (Locally and then in the cloud)._ The code follows a similar pattern and can be found in  [github](https://github.com/ashhadulislam/medium_weedVcrop-main)  and the running app can be found  [here](https://ashhadulislam-medium-weedvcrop-main-main-ppo37r.streamlit.app/).

First we have shown how a single image can be processed by different yolo models and the corresponding results as shown in the gif below.

![](https://miro.medium.com/max/1200/1*YJFfygi_4JR5fDdKNIhoEQ.gif)

Single image processed by different yolov8 models

However, you might need to upload a set of images and apply the models on them. In that case, it would be tedious to drag and drop every image one by one. Rather, we have another page in the same application where you can drag and drop a zip file containing multiple images.

![](https://miro.medium.com/max/1200/1*0BDHmo-iISYYQWNHTa9qfg.gif)

Processing zipped files

You can even choose the from a list of models (yolov8 — nano, small, medium and large). You will get a zipped file containing folders corresponding to each model with a set of result in each.