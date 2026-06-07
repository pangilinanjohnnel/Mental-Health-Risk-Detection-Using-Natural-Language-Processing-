# Mental Health Risk Detection Using NLP

An automated system capable of detecting early signs of mental health issues (anxiety, depression, distress) through text analysis using a fine-tuned DistilBERT architecture.

## 📌 Problem Statement
Social media and online forums have become primary outlets for individuals to express distress (Guntuku et al., 2017). However, manually identifying these signals at scale is incredibly difficult and time-consuming. 

Prior studies demonstrate that linguistic cues serve as reliable indicators of psychological states (De Choudhury et al., 2013). There is a critical need for automated systems to detect these early signs and support researchers and mental health professionals.

## 🎯 Objectives
* **General Objective:** Develop a Natural Language Processing (NLP) model that classifies text into binary risk categories:
  * `0 (Normal)`: No indication of mental health distress.
  * `1 (At-Risk)`: Shows signs of anxiety, depression, or other mental health issues.
* **Specific Objectives:**
  * Analyze textual data for patterns associated with mental health conditions.
  * Provide a foundational tool to support early detection and monitoring systems.

## 📊 Dataset
* **Source:** [Mental Health Corpus (Kaggle)](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus) 
* **Original Size:** 27,977 entries
* **Sample Used for This Project:** 1,000 entries
* **Description:** Text data containing statements from individuals expressing various emotions, mental health struggles, and general day-to-day sentiments.

## 🛠️ Methodology & Architecture
* **Task:** Binary Text Classification
* **Model:** `DistilBERT` (Transformer-based architecture)
* **Rationale:** Chosen because it is lightweight, trains faster, and retains 95% of BERT's language understanding capabilities with a fraction of the computational overhead.

## 📈 Results / Evaluation
* Model has an overall accuracy of **88%**, which is quite good.  
* Shows slightly better recall for **class 1 at 0.92** and slightly better precision for **class 0 at 0.91**.  
* The **F1-scores are balanced** for both classes at **0.88**, suggesting a robust performance. 
* Backed up with a support of **102 for class 0** and **98 for class 1**—overall, a pretty balanced model performance.

## Conclusion: 
The study successfully developed an NLP-based classification model capable of detecting mental 
health risk from textual data. With an accuracy of 88% and strong recall for at-risk cases, the model 
demonstrates practical potential in assisting mental health monitoring systems.

## Reflection: 
The model performed well despite using a relatively small dataset of 1000 samples, more data 
could further improve accuracy and generalization of the model. Furthermore, possible multi
classification can be applied for future version of this study.

## Notes: 
Random state is applied so upon further testing accuracy of 88% can still go higher 
