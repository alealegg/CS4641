# Predicting Whether a Patient Has a Brain Tumor From a Brain Scan

**Team Members:** Alea Legg, Brennan Oconnor, Mark Wetherly

## Presentation
https://youtu.be/xFP-dhI4ev0

## Proposal 

![Info](https://user-images.githubusercontent.com/31289084/121968442-573abb80-cd40-11eb-89e1-f002ce4db711.png)

### Introduction/Background
We aim to improve accuracy and reduce costs associated with human inspection by using machine learning to identify tumors in brain scans of patients. Currently doctors have to manually classify scans. By creating a machine learning algorithm that can correctly label images with tumors, we can quickly and accurately analyze large numbers of brain scans, reducing demands on doctors whose time could be spent better assisting patients. Our goal is to formulate an algorithm that can identify tumors in patients faster and with more accuracy than humans. Our algorithm should be able to correctly identify tumors for a diverse set of patients of different ages, and tumors in various stages of growth.

### Methods
Our dataset contains 4600 grayscale images of X-ray scans of the human brain. The data is labelled, and 55% of the images are classified as having a tumor present and the remaining 45% are classified as healthy with no tumor present. We plan to use the techniques of image segmentation algorithms, for example Gray Level Co-occurrence matrix, to group pixels showing abnormalities that could be a tumor as well as soft clustering algorithms, such as Fuzzy C-Means, to assign pixels to groups for a probabilistic model. These methods are based on research from our references.

### Results
We hope to develop a model that analyzes images to correctly indicate the presence of tumors. Binary classification will indicate whether a tumor is present or not. Detailed results can be seen under the Unsupervised Learning and Supervised Learning sections below.

### Discussion
If a model could accurately determine the presence of brain tumors, healthcare workflows would experience significant boosts in speed and efficiency. With a high volume
of scans entering the pipeline every day, radiologists and other medical professionals often find themselves overwhelmed, especially in situations where budget cuts 
lead to understaffing. An AI specializing in brain scans would certainly alleviate bottlenecks that occur in the diagnostic process, primarily through the ability to
produce potentially quicker and more accurate diagnoses than a certified doctor could. Consequently, this would expand doctors' bandwidth and allow them more time to
focus on other tasks, including direct patient care. 

Perhaps most importantly, a successful implementation of such an AI would encourage healthcare experts to explore future applications of machine learning that could
improve diagnostic medicine. For instance, an upgraded version of this project's AI might identify tumor types and suggest recommended treatment options based on the
tumor type. To prevent machine learning from completely eliminating current jobs, AIs could simply be designed to analyze doctors' reports and flag any where the diagnosis
doesn't match the AI's diagnosis. Doing this before delivering results to patients would save time, money, and resources that would otherwise go towards handling false
diagnoses.

For discussion on the results we gathered, see the below sections for Unsupervised and Supervised Learning.

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

To convert our images to usable data, we did the following: 
* Removed any files that are not square, RGB, JPEG images (1598 image files remained in the data set)
* Converted the remaining images from RGB to grayscale and resized them to all be the same number of pixels using the skimage module
* Created a dataframe where each column represented a pixel, each row represented an image, and each value represented that image's pixel's intensity 

Once we cleaned and organized our data, we ran unsupervised machine learning techniques. We applied clustering algorithms with the goal of clustering the data into two groups (ideally one that represented healthy scans and one that represented brain tumor scans). 

### Clustering with K-Means
We applied the K-Means algorithm using the sklearn.cluster module where n = 2 clusters because ideally the model would group the images into two categories: healthy scans and brain tumor scans. We also applied Principle Component Analysis (PCA) to reduce the dimensionality of the data. Since the data has a very large set of features (one feature for each pixel), PCA allows us to reduce the number of features while still maintaining most of the information that the features provide. PCA also allowed us to visualize the images in their clusters. 

![kmeans](https://user-images.githubusercontent.com/31289084/125023981-ec825480-e04d-11eb-8111-561aed4631a2.png)

In the above image, the orange segment of the visual is clearly more densely populated with data. Since our current cleaned data is mainly made up of cancerous scans, it is possible that the model may have had a slight bias during clustering analysis. To improve the overall clustering, we should incorporate a greater volume of "healthy" images in our cleaned data. Doing so may improve the model's performance and allow us to see the two clusters more definitively. Currently 615 of the 1599 "cleaned" images (38%) are of non-cancerous (healthy) scans.

### Clustering with DBSCAN
We also performed a preliminary application of the DBSCAN algorithm to see if we could gain more insight on our results. 

![Elbow Method for eps](https://user-images.githubusercontent.com/84369101/125132988-28133200-e0d3-11eb-8309-91a84eb56fd3.png)


After plotting the elbow method, we used eps = 2.5 as a rough estimate of the ideal neighborhood parameter. Since the data is two-dimensional we opted for min_pts = 2x2 = 4. 

![DBSCAN Nonfiltered Images](https://user-images.githubusercontent.com/84369101/125132312-25fca380-e0d2-11eb-812d-ff926b986643.png)

DBSCAN with these parameters generated 40 clusters, which we immediately knew was too many. Despite the excessive clusters, the large blue cluster appears to classify images with an acceptable initial degree of accuracy. To try and improve the performance of the DBSCAN algorithm, we filtered the images to reduce noise in the data (discussed below) and used the optimal eps, which is equal to 2.1 (also discussed below). 

### Applying Filters to Reduce Noise
The lack of accuracy in the results of the above clustering algorithms show that our images probably contain a lot of noise. We applied a filter to reduce this noise in the hopes of attaining more accurate clustering. Based on some research, MRI images are prone to Gaussian noise, and a bilateral filter is one type of filter than can reduce this type of noise. After appling the filter to the images and running the same algorithms as above, we obtained the following results: 

![kmeans_filtered](https://user-images.githubusercontent.com/31289084/125023785-90b7cb80-e04d-11eb-95b3-2a414e604983.png)

![dbscan_filtered](https://user-images.githubusercontent.com/31289084/125023930-d2e10d00-e04d-11eb-99e5-ab07315bc27f.png)

While the filter did not make any major changes to the K-Means clusters, there was a change to the DBSCAN clusters. When using the optimal epsilon for the filtered images, the DBSCAN algorithm clustered the images into 21 groups. While this is still much larger than the ideal 2 groups, it is a major improvement from the first DBSCAN algorithm (run on the images before applying the filter) which clustered the images into 40 groups. So even though there is still noise that prevents the model from being accurate, some noise was removed from the original images which improved the accuracy compared to the first DBSCAN clusters. Some other steps that could be taken to reduce noise might include applying more filters and/or removing outliers. 

Note: The images the filters were applied to were resized to (200,200) instead of (400,400) like above to reduce the time it took to run the code. This may have resulted in some loss of information for some images but it overall improved the model. 


### Resources and References Used
* https://aidancoco.medium.com/data-cleaning-for-image-classification-de9439ac1075
* https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/, 
* https://note.nkmk.me/en/python-numpy-image-processing/
* https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html?highlight=dbscan
* https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
* https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_bilateral
* https://medium.com/image-vision/noise-filtering-in-digital-image-processing-d12b5266847c
* https://www.intechopen.com/books/high-resolution-neuroimaging-basic-physical-principles-and-clinical-applications/mri-medical-image-denoising-by-fundamental-filters
* https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
* https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
