# LengthOfStayANN
Due to the intensive treatment process of coronavirus pneumonia cases, it is important to predict the Length of Stay (LOS) of patients at the hospital to allow better management of resources and increase the efficiency of hospital services to provide improved healthcare. To predict LOS, we used four artificial neural network models namely the Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Multilayer Perceptron with PCA (PCA+MLP), and the Bidirectional Long Short Term Memory (BiLSTM) model to analyze the advantages and disadvantages of the different models using the Microsoft Hospital Length of Stay data. The proposed method is compared with the state-of-the-art models and a simple MLP model. Our models achieved an accuracy between 73% and 88% with the CNN model providing the highest accuracy.

Keywords — ANNs, CNN, LSTM, COVID-19, Corona, hospital length of stay, accuracy, predictive analytics

# Publication
[Z. Fu, X. Gu, J. Fu, M. Moattari and F. Zulkernine, "Predicting the Length of Stay of Patients in Hospitals," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 3150-3156, doi: 10.1109/BIBM52615.2021.9669527](https://ieeexplore.ieee.org/document/9669527)

# Dataset
The LOS dataset used in this study was published by Microsoft in 2017. The LOS dataset contains 100,000 records of COVID-19 patients’ information collected from multiple local hospitals. Each line of the LOS dataset has 27 features representing detailed information about each patient including gender, number of readmissions in the past 180 days, and a text note indicating whether the patient was diagnosed with other illness during the hospitalization such as psychological disorder, renal disease, asthma etc. 57% of the records are of female and 43% of male subjects. Each record is uniquely identified by a hospital admission number (ID) and an encounter number or ID (from 1 to 100,000). The hospital admission IDs represent different hospital locations, and each hospital has a different capacity to admit patients. Most biometric measurements taken during hospitalization are recorded as floating-point numbers. The “length of stay” is recorded as a number ranging from 1 to 17 days. We train our models to predict this value and compare the predicted value with this label to compute the accuracy

# Note
We have also examined the data in statistical models, ex: XGB, Random Forest, and LGBM. 
Grid Search, Random Search, and Bayesian Search were used to find the proper parameters of each statistical model. 
