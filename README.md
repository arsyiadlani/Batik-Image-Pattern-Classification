# Popular Batik Pattern Image Classification
English | Indonesian

## A. Project Background
Becoming a maritime nation consisting of 17,000 islands and enjoyed by 272,229,372 inhabitants, a surprise when Indonesia is known as a country with extraordinary ethnic, religious, linguistic and cultural diversity. This diversity, especially culture, is a noble thing for society Indonesia because it can manifest in diversity—Bhinneka Tunggal Ika—to become a value that is felt real in socialize. Even so, there are not many Indonesian people who appreciate and recognize each existing culture. Batik, as one of the cultural products from the island of Java, also have various patterns which later developed into an icon that is not limited only to the island of Java. It is a relatively difficult thing to know (and memorize) every types of batik because of various varieties that it has, so that it becomes a a challenge in itself in the conservation of Indonesian culture. 

Identification of batik patterns becomes an interesting problem, which goes hand in hand the development of civilization today, technology can be a solution the problem. Machine learning (ML) is one of the breakthroughs today's technology, with various uses and benefits. One of the branch of Machine Learning classification is, where the algorithm reads input in the form of existing data that is classified into several groups for later predict in the new set of input data should be. In this project, we will be explore various methods using standard model architecture such as Inception and also using self-built CNN model architecture to find the best method to carry out the task to classify the most popular Batik patterns, namely Kawung, Parang, Truntum, Mega Mendung, and combined patterns.

## B. Project Description
For data collection process, we scraped online images of each individual pattern automatically with total of 1400 images. Then, data preprocessing for training data is carried out, which prepared into two different training datasets. Training data training_files_A is prepared only by renaming and regrouping treatment, while training_files_B is prepared through various preprocessing treatments, including zooming, resizing, shearing, and cropping. The test_files dataset is prepared manually containing around 50 images with known ground truth label to test and compare the trained model performance. For the training process, we used two methods, namely the existing InceptionV3 standard model, and also the self-built CNN architecture. Each method will be trained on both training_files_A and training_files_B and will be tested using test_files to compare the performance results.

- training_files_A: https://drive.google.com/drive/folders/1Z0xF4HhIUSmiUvHXfGw9fqJMYF6291Zs?usp=sharing
- training_files_B: https://drive.google.com/drive/folders/1Ios-4RnDemzmD1bnfGBnIWqE1UcvBnGP?usp=sharing
- test_files: https://drive.google.com/drive/folders/10pVKHX9n5TFj54iNh3l07XS5cottMdrl?usp=sharing

## C. Results
In the classification experiment using InceptionV3 model, it was found that testing test data on programs trained using training_files_A produces an accuracy of 99%. Then, when running the test on program trained using training_files_B, performance is experiencing decreased to 94%. The same pattern also occurs in the development model which using a self-built CNN model architecture. This phenomenon is likely due to the augmentation process the data carried out causes the variability of the data to be much greater thus requiring a higher level of model complexity to training_files_B.

## D. Appendix
- a-train.py: Train the model with training_files_A using standard InceptionV3 model architecture
- a-action.py: Test the performance of the trained model with test_files using standard InceptionV3 model architecture
- b-train.py: Train the model with training_files_B using standard InceptionV3 model architecture
- b-action.py: Test the performance of the trained model with test_files using standard InceptionV3 model architecture
- c-train.py: Train the model with training_files_A using self-built CNN model architecture
- c-action.py: Test the performance of the trained model with test_files using self-built CNN model architecture
- d-train.py: Train the model with training_files_B using self-built CNN model architecture
- d-action.py: Test the performance of the trained model with test_files using self-built CNN model architecture
