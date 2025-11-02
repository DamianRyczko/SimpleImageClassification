# SimpleImageClassification
Here i implemenrted 2 models representing 2 different aproaches to the problem. 

# How to Run
* First download the [dataset](https://www.kaggle.com/datasets/gergvincze/simple-hand-drawn-and-digitized-images/data) and place it in 'data' folder or place it whenever you like but change path so it points to folder with dataset. 
* To run the notebook, you first need to install the libraries from the requirements.txt file.
* The notebook contains two models, representing two different approaches to the problem.
* The models trained in the notebook have a dedicated web application built in Streamlit (in the ImageRecognitionWebiste folder).
* To launch the app, navigate to the main.py file's directory within your Python virtual environment and run the following command:
### streamlit run main.py
* After executing the command, the application will automatically open in your web browser.

A note on performance: I ran this notebook in an Ubuntu WSL environment using a GPU, and it consumed a significant amount of RAM. This was especially true for the "basic" model (the one built from scratch). If you have trouble running this model, the transfer-learning model (based on Google's MobileNet) should work fine, train much faster, and is the better model overall.
      
# My Conclusions
* The dataset is quite small, and the biggest challenge throughout this project was overfitting. This was successfully managed using Dropout and L2 Regularization.
* Without Data Augmentation, the model is extremely brittle. It doesn't learn what a "bicycle" is in general; it only learns to recognize the specific bicycles from the dataset.
* The small dataset means the model might fail to learn certain features if all examples of that feature randomly end up in the validation set. Using StratifiedKFold cross-validation helps solve this.
* Even with Data Augmentation, there is a clear performance ceiling. While more distortion generally leads to better generalization, applying transformations that are too aggressive prevents the model from correctly identifying the classes.
* The small dataset also results in a small validation set. This causes the Accuracy and Loss vs. Epochs charts to be very "jagged" or noisy.
* Because of this, the primary metric for model evaluation was the confusion matrix.
* However, the validation charts were still very useful for identifying issues like underfitting or overfitting in real-time.
# Potential Improvements
* A better base model for transfer learning could certainly have been chosen. However, I selected MobileNetV2 because it is only 10MB, and I wanted to ensure the final project was small enough to be easily uploaded to GitHub.
* In the future, working with a larger dataset would be ideal. This would not only improve model learning but also largely resolve the "brittle model" problem.
      
