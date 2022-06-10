# Garbage Classifier: Machine Learning

## Machine Learning Team:
### Meet the team:
   1. Zenix Fajar Luthfiansyah (M2008Q0853) - [Github profil](https://github.com/zenyxfajar)
        - Pre-Processing data
        - Build model
        - Train and test model
        - Model deployment
   2. Yeni Rismawati (M2227G2097) - [Github profil](https://github.com/yenirsmwati)
        - Do what 

### Dataset used in this project:
   - In this project we use a [GARBAGE DATASET](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) from kaggle.com, which is classified into six classes : 
      - Plastic (label 0)
      - Metal (label 1)
      - Paper (label 2)
      - Cardboard (label 3)
      - Glass (label 4)
      - Waste (label 5)
   - The total of this dataset is 5123 images of garbage which are divided into : 
      - Train : 80%
      - Test : 10%
      - Validation : 10%

### The workflow (alur kerja)
   1. Input the images of garbage
   2. The model will be processed, and classify the input image into one of the classes (**Plastic, Metal, Paper, Cardboard, Glass, Waste**)
   3. The output is a classification result from the input image. The result is the class that has the highest probability value.

### Transfer learning used
   - The model that we used is transfer learning architecture : `MobileNet V2`. The architecture of `MobileNet V2` has a lightweight and implementable performance on mobile devices. In this model, the loss used is `categorical cross-entropy` because we use multiclass classification. And in this model, the best optimizer is `Adam`.
- train model use --------
- for deployment -----

### Output
