# Text Classification - Capstone Project

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

## Baseline Model

Using Logistic Regression as baseline model, we see that accuracy is at 28%

## Multiple models tested to compare accuracy before sampling. 

| Model                     | Accuracy  |
|--------------------------|-----------|
| Logistic Regression       | 0.2757    |
| Multinomial Naive Bayes   | 0.3773    |
| Support Vector Machine    | 0.1610    |
| Random Forest             | 0.0865    |

### Interpretation:
All models perform poorly, with accuracies below 40%. This suggests severe class imbalance, where the models are likely biased toward predicting the majority class and are not learning meaningful patterns from the minority classes.

## Multiple Models using SMOTE sampling.

![image](https://github.com/user-attachments/assets/1e318900-7389-4abc-b4ad-231451a1d2bc)

| Model                     | Accuracy  |
|--------------------------|-----------|
| Logistic Regression       | 0.2284    |
| Multinomial Naive Bayes   | 0.3219    |
| Support Vector Machine    | 0.1308    |
| Random Forest             | 0.0785    |

### Interpretation:
Surprisingly, SMOTE did not improve performance—in fact, accuracies dropped in most models. This could be due to:

SMOTE generating synthetic examples that don't generalize well for these models.

Models struggling with overfitting to synthetic samples that don't capture the true distribution.

## Multiple Models using Random Over sampling.

![image](https://github.com/user-attachments/assets/de67783e-ace4-40f7-a15e-bb6994dc10fe)

| Model                     | Accuracy  |
|--------------------------|-----------|
| Logistic Regression       | 0.7931    |
| Multinomial Naive Bayes   | 0.7798    |
| Support Vector Machine    | 0.7943    |
| Random Forest             | 0.7957    |

### Interpretation:
Dramatic improvement across all models, especially for Random Forest (from 0.0865 to 0.7957). This suggests:

Random oversampling was effective in balancing the dataset without introducing synthetic noise.

The models were able to learn much better when each class had enough genuine samples (even if repeated).

Random Forest benefited most, possibly due to its robustness and ability to handle high-variance data well.

## Transformer Model: Bio_ClinicalBERT

### Model
We used emilyalsentzer/Bio_ClinicalBERT, a domain-specific BERT model pretrained on clinical notes and biomedical literature, to test whether it could outperform traditional models in classifying medical specialties.

The goal was to leverage its understanding of clinical language to improve accuracy on transcription data, especially given the complex medical terminology involved. We expected that its contextual embeddings would capture more meaning than traditional TF-IDF features.

However, despite its domain-specific training, Bio_ClinicalBERT underperformed compared to traditional models enhanced by random oversampling. This suggests that data balance and task-specific tuning play a more critical role than model complexity alone in this context.


---
### Training Progress

| Step     | Training Loss |
|----------|----------------|
| 500      | 3.4449         |
| 1000     | 2.9984         |
| Final    | 2.9559         |

TrainOutput(
global_step=1479,
training_loss=2.9559,
metrics={
'train_runtime': 8024.14,
'train_samples_per_second': 1.474,
'train_steps_per_second': 0.184,
'epoch': 3.0
}
)


---

### Evaluation Metrics

- **Validation Accuracy**: `0.3256`
- **Validation F1 Score**: `0.2482`

Despite being domain-specific, Bio_ClinicalBERT did not outperform traditional models enhanced by random oversampling.

---

### Saving and Loading Fine-Tuned BERT Model

The fine-tuned BERT model and tokenizer were successfully saved to disk to enable reuse without the need for retraining. This ensures efficient deployment and scalability for inference tasks. The saved model and tokenizer were later loaded to perform predictions on new and unseen data. This process confirmed that the model retained its fine-tuned performance and was able to make accurate predictions consistently.

---

## Conclusion

- Traditional models struggled with imbalanced data initially.
- **SMOTE** did not help; it often worsened performance.
- **Random Over-Sampling** significantly improved all models' accuracy.
- **Bio_ClinicalBERT** underperformed compared to traditional models after oversampling:
  - Accuracy: 0.3256
  - F1 Score: 0.2482
- **Oversampling plus traditional models** performed better than the transformer model without balancing, in this case.




