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

https://user-images.githubusercontent.com/57211163/214333702-7a666c70-3499-470d-8b39-fc37374950ab.mp4

Once an app is loaded, 

1. Upload an image of food.
2. Once the image is processed, **`Predict`** button appears. Click it.
3. Once you click the **`Predict`** button, the model prediction takes place and the output will be displayed along with the model's **Top-5 Predictions**
4. And voilà, there you go.


## Model Training 

> If you actually want to know the Nuts and Bolts how the model was trained check out **[`FoodVisionGithub.ipynb`](https://github.com/faroukbrachemi/FoodVision/blob/main/foodvisiongithub.ipynb) Notebook**

1. #### Imported Food101 dataset from **[Tensorflow Datasets](https://www.tensorflow.org/datasets)** Module.

2. #### Becoming one with the Data :

3. #### Setup Global dtype policy to **`mixed_float16`** to implement [**Mixed Precision Training**](https://www.tensorflow.org/guide/mixed_precision)

   > Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it **run faster** and use **less memory**.

4. #### Building the Model Callbacks 

   As we are dealing with a complex Neural Network (EfficientNetB0) its a good practice to have few callbacks set up. Few ones I used throughtout this Notebook are :

   - **TensorBoard Callback :** TensorBoard provides the visualization and tooling needed for machine learning experimentation

   - **EarlyStoppingCallback :** Used to stop training when a monitored metric has stopped improving.

   - **ReduceLROnPlateau :** Reduce learning rate when a metric has stopped improving.


5. #### Built a  [Fine Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)  Model

   **Architecture** : **`EffficientNetB1`**

6. #### Evaluating and Deploying out Model to Streamlit

   Once we have our model ready, its cruicial to evaluate it on our **custom data** : *the data our model has never seen*.

   Training and evaluating a model on train and test data is cool, but making predictions on our own realtime images is another level.

   Once we are satisfied with the results, we can export the model as a `.hdf5`  which can be used in future for model deployment.

Once the model is exported then there comes the Deployment part. Check out  **[`app.py`](https://github.com/gauravreddy08/food-vision/blob/main/food-vision/app.py)** to get more insight on How I integrated it with Streamlit.

## Breaking down the repo

At first glance the files in the repo may look intimidating and overwhelming. To avoid that, here is a quick guide :

* `.gitignore` : tells what files/folders to ignore when committing
* `app.py`  : Our Food Vision app built using Streamlit
* `utils.py`  : Some of used fuctions in  `app.py`
* `foodvisiongithub.ipynb`  :  Notebook used to train the model
* `model/`  : Contains all the models used as *.hfd5* files
* `requirements.txt`  : List of required dependencies required to run `app.py`
* `extras/`  : contains images and files used to write this README File


######                                             *Inspired by **[Daniel Bourke's Course](https://github.com/mrdbourke/tensorflow-deep-learning/)** *

