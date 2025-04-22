# Text Classification - Text Classification capstone project

## Project Goal:

The primary objective of this project is to develop a machine learning model capable of accurately predicting the medical specialty associated with a given medical transcription. While a publicly available dataset is being used for initial development and learning the process, the final models will be applied to classified company data that cannot be publicly shared. This ensures privacy compliance while supporting internal implementation. 

Link to Jupiter notebook: https://github.com/MRiDeb/Text-Classification/blob/main/Text_Classification1.ipynb

## Business Objective:

With publicly available medical transcription data, the goal is to classify the medical specialty based on the content of the transcription. This allows for safe experimentation and development of NLP techniques, helping to refine the classification pipeline and model performance in a non-sensitive setting.

## Exploratory Data Analysis (EDA):





### Data Cleaning:
For this project, only the transcription and medical_specialty columns are retained. All other columns are removed. The dataset contains 33 missing values in the medical_specialty column, which have been filled with the label 'unknown'.

### Data Visualization

Visualize the distribution of specialties. From this graph, it is clear that some specialties have very limited data while others have significantly more. The data is not evenly distributed. Techniques like oversampling or undersampling may be applied based on these findings.

![image](https://github.com/user-attachments/assets/a706adf0-48e3-4341-90cd-f37ef5814762)

### Text Preprocessing:
Initial steps included lowercasing, removal of punctuation, and elimination of stop words. Tokenization and lemmatization have been successfully implemented using the Punkt tokenizer.

### Feature Extraction
Convert text into numerical format using TF-IDF Vectorization.

### Baseline Model
Using Logistic Regression as baseline model, we see that accuracry is at 30%

### Advanced Models to boost accuracy

Naive Bayes

Random Forest

XGBoost

BERT with HuggingFace Transformers -  This is failing, may need help with this. 

We'll also compare results using accuracy, F1-score, precision, and recall.
