# Garbage Classifier: Machine Learning

## Machine Learning Team:
### Meet the team:
   1. Zenix Fajar Luthfiansyah (M2008Q0853) - [Github profile](https://github.com/zenyxfajar)
        - Build model
        - Train and test model
        - Model deployment
   2. Yeni Rismawati (M2227G2097) - [Github profile](https://github.com/yenirsmwati)
        - Acquiring dataset
        - Analyzing dataset
        - Currating additional dataset
### Dataset used in this project:
   -  [Garbage dataset 1](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
   -  [Garbage dataset 2](https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy/data)
   -  The dataset used consists of 6 types of waste: **Cardboard, Glass, Metal, Paper, Plastic, and Trash**. The data will be divided into **Train, Test, and Validation** data, with 80% for Train, 10% for a test, and 10% for Validation. For the class label: 
      - Cardboard (Label 0)
      - Glass (Label 1)
      - Metal (Label 2)
      - Paper (Label 3)
      - Plastic (Label 4)
      - Trash (Label 5)
### The workflow (alur kerja)
   1. Input the images of garbage
   2. The model will be processed, and classify the input image into one of the classes (**Cardboard, Glass, Metal, Paper, Plastic, Trash**.)
   3. The output is a classification result from the input image. The result is the class that has the highest probability value.

### Transfer learning used
   - The model that we used is transfer learning architecture : `MobileNet V2`. The architecture of `MobileNet V2` has a lightweight and implementable performance on mobile devices. In this model, the loss used is `categorical cross-entropy` because we use multiclass classification. And in this model, the best optimizer is `Adam`.
   - We train this model with `batch size: 16`, `epoch: 15` and we get more than 90% `accuracy` for Training and Validation.
### Deployment
   - For deployment, we used `RESTful API` with `Flask-framework`. The method that we use is `GET` for get the data sent by user using the `POST` request and `POST` for requesting change in source, also can be used for sending data, in this case we use `POST` to send image data to be processed by application.
   - Application use Cloud Run API to run the application.
   - API : https://dopredict-uzgmpvvmea-uc.a.run.app

### Output
   - Build accuracy
   
   ![val and test acc 1](https://user-images.githubusercontent.com/101339523/173275671-04024887-cd35-4481-9a7d-5ca7b9c63205.jpg)

   - Test by sending `POST` request to Cloud Run API of application
   ![hasil deploy local](https://user-images.githubusercontent.com/101339523/173274358-65574ea7-cccd-4a6c-9c79-f6aefad78d73.jpg)
