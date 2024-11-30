# Food Vision :hamburger: :camera:



FoodVision is a **CNN Image Classification Model** that classifies images into **101 food** categories using the **Food101 dataset**. 
It can identify 101 different food classes.

It is based upon a pre-trained Model that has been fine-tuned on the **Food101 Dataset**.

### Fun Fact 

The Model actually beats the [**DeepFood**](https://arxiv.org/pdf/1606.05675.pdf) Paper's model which also trained on the same dataset.

The Accuracy aquired by DeepFood was **77.4%** and our model's **85%** . Difference of **8%** ain't much, but the interesting thing is, DeepFood's model took **2-3 days** to train while our's barely took **90min**.

> ##### **Dataset used :**  **`Food101`**

> ##### **Model Used :** **`EfficientNetB1`**

> ##### **Accuracy :** **`85%`**

### Check the [deployed app](https://foodvision-ledafarouk.streamlit.app/)

https://github.com/user-attachments/assets/c3256fad-5742-44da-9d88-dbb05afc910d


Once an app is loaded, 

1. Upload an image of food.
2. Once the image is processed, **`Predict`** button appears. Click it.
3. Once you click the **`Predict`** button, the model prediction takes place and the output will be displayed along with the model's **Top-5 Predictions**
4. And voilà, there you go.


## Model Training 

> If you want to know how the model was trained check out **[`FoodVisionGithub.ipynb`](https://github.com/faroukbrachemi/FoodVision/blob/main/foodvisiongithub.ipynb) Notebook**


## Breaking down the repo

At first glance the files in the repo may look intimidating and overwhelming. To avoid that, here is a quick guide :

* `.gitignore` : tells what files/folders to ignore when committing
* `app.py`  : Our Food Vision app built using Streamlit
* `utils.py`  : Some of used fuctions in  `app.py`
* `foodvisiongithub.ipynb`  :  Notebook used to train the model
* `model/`  : Contains all the models used as *.hfd5* files
* `requirements.txt`  : List of required dependencies required to run `app.py`
* `extras/`  : contains images and files used to write this README File


######                                             *Inspired by* **[Daniel Bourke's Course](https://github.com/mrdbourke/tensorflow-deep-learning/)**

