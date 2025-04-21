# Text Classification - Text Classification capstone project

Project Goal:
The primary objective of this project is to develop a machine learning model capable of accurately predicting the medical specialty associated with a given medical transcription. 

Classifier Models Under Consideration:

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (SVM)

Random Forest

Transformer-based models (e.g., BERT)

Business Objective:
The goal from a business perspective is to automate the classification of medical specialties based on transcription content. This automation improves clinical workflow efficiency, facilitates better care coordination by routing information to the appropriate specialists, and enables more effective resource allocation and operational planning. Ultimately, it contributes to improved patient outcomes and a reduced administrative workload.

Exploratory Data Analysis (EDA):

Data Cleaning:
For this project, only the transcription and medical_specialty columns are retained. All other columns are removed. The dataset contains 33 missing values in the medical_specialty column, which have been filled with the label 'unknown'.

Text Preprocessing:
Initial steps included lowercasing, removal of punctuation, and elimination of stop words. Tokenization and lemmatization are planned, but implementation is currently pending due to issues with the Punkt tokenizer download. This is being addressed.

Data Visualization:

Distribution of Medical Specialties: Helps detect class imbalance. Techniques like oversampling or undersampling may be applied based on these findings.

Transcription Length by Specialty: Identifies which specialties tend to generate longer or more detailed notes, informing preprocessing and model input strategy.

Top TF-IDF Terms per Specialty: Highlights key terms relevant to each specialty for improved feature extraction.

Top 10 Most Common Words per Specialty: Assists in identifying specialty-specific vocabulary.

Word Cloud of Transcriptions: Provides a visual summary of frequently occurring terms across all specialties.

Top 10 Most Frequent Words by Specialty: Reveals patterns in language use unique to each specialty, aiding model accuracy.

Feature Engineering:

Extract specialty-specific terms using Clinical BERT embeddings.

Apply Named Entity Recognition (NER) to identify and remove sensitive or irrelevant entities such as patient age and sex from the transcriptions.

Initial Modeling:
A preliminary model using the Multinomial Naive Bayes classifier was trained, achieving an accuracy of approximately 37%. Further tuning and exploration of more advanced models are planned in the next development phase.
