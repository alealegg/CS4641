# Predicting Whether a Patient Has a Brain Tumor From a Brain Scan

**Team Members:** Alea Legg, Brennan Oconnor, Mark Wetherly

## Proposal 

![Info](https://user-images.githubusercontent.com/31289084/121968442-573abb80-cd40-11eb-89e1-f002ce4db711.png)

### Introduction/Background
We aim to improve accuracy and reduce costs associated with human inspection by using machine learning to identify tumors in brain scans of patients. Currently doctors have to manually classify scans. By creating a machine learning algorithm that can correctly label images with tumors, we can quickly and accurately analyze large numbers of brain scans, reducing demands on doctors whose time could be spent better assisting patients. Our goal is to formulate an algorithm that can identify tumors in patients faster and with more accuracy than humans. Our algorithm should be able to correctly identify tumors for a diverse set of patients of different ages, and tumors in various stages of growth.

### Methods
Our dataset contains 4600 grayscale images of X-ray scans of the human brain. The data is labelled, and 55% of the images are classified as having a tumor present and the remaining 45% are classified as healthy with no tumor present. We plan to use the techniques of image segmentation algorithms, for example Gray Level Co-occurrence matrix, to group pixels showing abnormalities that could be a tumor as well as soft clustering algorithms, such as Fuzzy C-Means, to assign pixels to groups for a probabilistic model. These methods are based on research from our references.

### Results
We hope to develop a model that analyzes images to correctly indicate the presence of tumors. Binary classification will indicate whether a tumor is present or not.

### Discussion
If a model could accurately determine the presence of brain tumors, healthcare workflows would experience significant boosts in speed and efficiency. With a high volume
of scans entering the pipeline every day, radiologists and other medical professionals often find themselves overwhelmed, especially in situations where budget cuts 
lead to understaffing. An AI specializing in brain scans would certainly alleviate bottlenecks that occur in the diagnostic process, primarily through the ability to
produce potentially quicker and more accurate diagnoses than a certified doctor could. Consequently, this would expand doctors' bandwidth and allow them more time to
focus on non-brain scans, knowing that the brain-related scans would be handled by an AI. 

Perhaps most importantly, a successful implementation of such an AI would encourage healthcare experts to explore future applications of machine learning that could
improve diagnostic medicine. For instance, an upgraded version of this project's AI might identify tumor types and suggest recommended treatment options based on the
tumor type. To prevent machine learning from completely eliminating current jobs, AIs could simply be designed to analyze doctors' reports and flag any where the diagnosis
doesn't match the AI's diagnosis. Doing this before delivering results to patients would save time, money, and resources that would otherwise go towards handling false
diagnoses.

### References
These literary references will help us decide which algorithms and models to use in our project.

1. Brain Tumor Detection based on Machine Learning Algorithms

    In this paper,  the researchers use machine learning algorithms to detect tumors in MRI images using a 3 step process, which includes preprocessing, feature extraction, and classification. In preprocessing,  the images are converted from RGB to gray scale and the noise is removed. In the feature extraction, the researchers used a gray level co-occurrence matrix (GLCM). 
   
    Background on GLCM : 
    
    Basically it calculates how often a pixel with a specific gray color value (aka tone) occurs in different directions adjacent to pixels with a different specific tone.           
    Based on this data, the researchers could then classify whether the tumor is benign or malignant. 
    
2. Conditional spatial fuzzy C-means clustering algorithm for segmentation of MRI images

    In this paper, the researchers discuss the fuzzy c-means (FCM) clustering algorithm and its role in image segmentation. 
    
    A bit more on FCM : 
    
    It is similar to K-means, however, in the case of FCM, a data point can belong to more than one cluster with a likelihood, and it is soft clustering. The researchers also use an upgraded version – conditional spatial fuzzy c-means (csFCM) in their research.
    
3. A Hybrid CNN- GLCM Classifier for Detection and Grade Classification of Brain Tumor

    In this recent paper, the researchers use an "image analysis scheme" called CNN (Convolutional Neural Network) Deep Net which classifies a tumor as benign or malignant. Basically CNN takes an input image, assigns weights to different objects in the image, and so now you can differentiate one object from another.

## Unsupervised Learning

### Data Cleaning:

To convert our images to usable data, we took the following steps: 
* Removed any files that are not sqare, RGB, JPEG images
* Converted the remaining images grayscale and resized them to be the same number of pixels using the skimage module
* Created a dataframe where each column represented a pixel, each row represented an image, and each value represented that image's pixel's intensity 

Once we cleaned our data, we were ready to run unsupervised Machine Learning techniques. 

### Machine Learning: 


