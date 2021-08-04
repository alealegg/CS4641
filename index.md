
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
We hope to develop a model that analyzes images to correctly indicate the presence of tumors. Binary classification will indicate whether a tumor is present or not. Results can be seen under the Unsupervised Learning and Supervised Learning sections below.

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

Further discussion and analysis can be seen under the Unsupervised and Supervised Learning sections below.

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
* Removed any files that are not square, RGB images (1598 image files remained in the data set)
* Converted the remaining images from RGB to grayscale and resized them to all be the same number of pixels using the skimage module
* Created a dataframe where each column represented a pixel, each row represented an image, and each value represented that image's pixel's intensity 

Once we cleaned and organized our data, we ran unsupervised machine learning techniques. We applied clustering algorithms with the goal of clustering the data into two groups (ideally one that represented healthy scans and one that represented brain tumor scans). 

### Clustering with K-Means
We applied the K-Means algorithm using the sklearn.cluster module where n = 2 clusters because ideally the model would group the images into two categories: healthy scans and brain tumor scans. We also applied Principle Component Analysis (PCA) to reduce the dimensionality of the data. Since the data has a very large set of features (one feature for each pixel), PCA allows us to reduce the number of features while still maintaining most of the information that the features provide. PCA also allowed us to visualize the images in their clusters. 

![Kmeans plot](https://user-images.githubusercontent.com/74310974/128102697-7046c8b1-056f-4860-bfac-47304ba74029.JPG)

In the above image, the blue segment of the visual is clearly more densely populated with data. White points indicate images with ground-truth "tumor" labels, while black points represent those images with ground-truth "healthy" labels. Since our current cleaned data is mainly made up of cancerous scans, it is possible that the model may have had a slight bias during clustering analysis. 615 of the 1598 "cleaned" images (38%) are of non-cancerous (healthy) scans.

### Clustering with DBSCAN
We also performed a preliminary application of the DBSCAN algorithm to see if we could gain more insight on our results. 

![Elbow Method for eps](https://user-images.githubusercontent.com/84369101/125132988-28133200-e0d3-11eb-8309-91a84eb56fd3.png)


After plotting the elbow method, we used eps = 2.5 as a rough estimate of the ideal neighborhood parameter. Since the data is two-dimensional we opted for min_pts = 2x2 = 4. 

![DBSCAN Nonfiltered Images](https://user-images.githubusercontent.com/84369101/125132312-25fca380-e0d2-11eb-812d-ff926b986643.png)

DBSCAN with these parameters generated 40 clusters, which we immediately knew was too many. Despite the excessive clusters, the large blue cluster appears to group images with an acceptable initial degree of accuracy. To try and improve the performance of the DBSCAN algorithm, we filtered the images to reduce noise in the data (discussed below) and used the optimal eps, which is equal to 2.1 (also discussed below). 

### Applying Filters to Reduce Noise
The lack of accuracy in the results of the above clustering algorithms shows that our images probably contain a lot of noise. We applied a filter to reduce this noise in the hopes of attaining more accurate clustering. Based on some research, MRI images are prone to Gaussian noise, and a bilateral filter is one type of filter than can reduce this type of noise. After appling the filter to the images and running the same algorithms as above, we obtained the following results: 

![Kmeans filtered](https://user-images.githubusercontent.com/74310974/128102709-b2ef5485-3845-45c6-b8cd-b6e1aed48666.JPG)

![dbscan_filtered](https://user-images.githubusercontent.com/31289084/125023930-d2e10d00-e04d-11eb-99e5-ab07315bc27f.png)

Although the K-Means clusters are tighter and have less spread, the filter did not make major improvements for K-Means. There was, however, a significant change to the DBSCAN clusters. When using the optimal epsilon for the filtered images, the DBSCAN algorithm clustered the images into 21 groups. While this is still much larger than the ideal 2 groups, it is a major improvement from the first DBSCAN algorithm (run on the images before applying the filter) which clustered the images into 40 groups. So even though there is still noise that prevents the model from being accurate, some noise was removed from the original images which improved the accuracy compared to the first DBSCAN clusters. Some other steps that could be taken to reduce noise might include applying more filters and/or removing outliers. 

Note: The images that the filters were applied to were resized to (200,200) instead of (400,400) like those above in order to reduce the time it took to run the code. This may have resulted in some loss of information for some images, but it overall improved the model. 

### Performance Metrics
We used several functions from sklearn to assess how effective our unsupervised learning algorithms performed.

K-means:
To analyze our results we used 3 metrics with output in the range [0,1], with 1 representing perfect clustering.

Completeness measures if all members of a given class are in the same cluster.
Homogeneity measures the degree to which clusters contain only data points that are members of a single class.
Normalized mutual info returns the interdependence of the predicted and true values.

* Completeness = 0.13
* Homogeneity = 0.12
* Normalized Mutual Info = 0.13

Based on the results we can conclude that our K-means clustering algorithm did not perform well. This is corroborated by the graphs with points labelled by ground truth showing poor clustering.

DBSCAN:
We used the Davies-Bouldin Score which measures the average similarity measure of each cluster with its most similar cluster. Ideally the value is 0.

Davies-Bouldin Score = 1.43

This score and our number of 21 clusters when ideally we would have only 2 indicates that the DBSCAN algorithm did not perform well for our dataset.


## Supervised Learning

### Data Adjustments
1. Data split into training and testing data: We randomly split our data into training and testing sets so that we would be able to test our trained models. 80% of our data was used for training, and 20% was used for testing. 
2. Addition of labels: Using the true labels of the images will allow us to test how accurate our models are by comparing the true/actual labels with the predicted labels. 

### Neural Network
We used an MLP Classifier from the sklearn neural network module to train and test a neural network classifer. After experimenting with different combinations of activation functions, solver types, and network shapes, we determined that a classifier using a relu activation function with lgbs solver and 3 hidden layers with 30, 20, and 10 nodes would best predict labels for the brain scans. Our final classifier predicted the labels with 98.2% accuracy. We also looked at the success rates for each class of images: the classifier predicted labels for images with no tumors with 99.2% accuracy and predicted labels for images with tumors with 97.6% accuracy. A total of six images were misclassified (only one non-cancerous image was misclassified). 

### SVM Classifier
We used an SVM Classifier from the sklearn svm module to train and test another classifier. This classifier predicted the labels with 97.3% accuracy. When looking at the accuracy of the classes individually, the success rate for labeling images with no tumor was 93.7%, and the success rate for labeling images with a tumor was 99.5%. A total of nine images were misclassified (only one cancerous image was misclassified). 

### Analysis

#### Comparison of Classifiers
Both classifiers performed with very high and very similar accuracy rates; the neural network only slightly outperformed the SVM classifier. The neural network classifier was better at predicting the labels for non-cancerous images, while the SVM classifier was better at predicting the labels for cancerous images. Overall, both were successful classifiers. 

<img width="361" alt="Accuracy" src="https://user-images.githubusercontent.com/31289084/128103578-078d1d99-f9a6-4f5f-9bb5-6371722fd42b.png">


#### Misclassification of Images
We took a look at a sample of images that were misclassified to see if we could understand why these scans might have not been labeled correctly by the classifiers. 

The following image was the only scan that was mislabeled by both classifiers: 

![image](https://user-images.githubusercontent.com/31289084/128103818-a4fe9d4a-13bd-4c0f-99ad-43a549d85dbe.png)

This scan is a non-cancerous image but both classifiers labeled it as cancerous. This scan appears to have been taken from the front or back of the head, and the majority of images in the dataset (and therefore a majority of images used to train the model) were scans taken of the top of the brain. This could explain why the classifiers did not label this scan correctly. Additionally, other parts of the head outside of the brain appear to be visible in this scan, and these parts could have been seen as tumors by the classifiers. For these reasons, it is not surprising that this scan was misclassified. 

The following images are cancerous scans that were mislabeled as non-cancerous by the neural network classifier:

![brain](https://user-images.githubusercontent.com/31289084/128104101-aae1f5c3-c3ae-4b12-9149-dc99980e76cc.png)
![brain2](https://user-images.githubusercontent.com/31289084/128104109-c0c7e284-57d0-4014-8e44-c91426500b33.png)
![brain3](https://user-images.githubusercontent.com/31289084/128104116-0bd7d7d4-4858-4d8a-a97b-6e6b20c146e6.png)
![brain4](https://user-images.githubusercontent.com/31289084/128104118-1c080522-ced6-4b31-989d-aab7c8afef35.png)

While it is surprising that the first and fourth scans were misclassified because there appears to be an obvious tumor, it is not surprising that the second scan was misclassified as there does not appear to be an obvious tumor. If it is hard for our human eyes to find a tumor, it is expected that it would also be hard for these classifiers to find a tumor. It is also not surprising that the third scan was misclassified because it is also a scan taken from a different view than the majority of the data (like the first scan discussed above). 

The following images are non-cancerous scans that were mislabeled as cancerous by the SVM classifier: 

![brain5](https://user-images.githubusercontent.com/31289084/128104127-4a2bc98f-742c-4d3d-8bdf-48256ca7aa5f.png)
![brain6](https://user-images.githubusercontent.com/31289084/128104147-0951111a-11fd-4d68-9f7c-a86172dbc2e3.png)
![brain7](https://user-images.githubusercontent.com/31289084/128104153-94aefcf3-0787-4085-9af4-864393f9fe52.png)
![brain8](https://user-images.githubusercontent.com/31289084/128104161-55c00174-8131-4804-aa73-a9574272b6f2.png)

While it is surprising that the first and third scans were misclassified because there does not appear to be anything noticeably different about these scans from the other non-cancerous scans, it is not surprising that the second and fourth scans were misclassified. These scans were taken from above like the majorty of the scans, but they also appear to include parts of the head outside of the brain, such as the eyeballs, that the classifiers most likely viewed as tumors. 

To improve the accuracy of our classifiers, only brain scans taken from above that only include the brain could be used to train and test the data. This would provide more consistency without confusing the classifiers. 

## Resources and References Used
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
