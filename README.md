<!-- HEADER BANNER -->
![Banner](https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:FF4500&height=200&section=header&text=Violent%20Crimes%20Per%20Population&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=35)

<!-- ANIMATED TAGLINE -->
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=FF4500&center=true&vCenter=true&width=800&lines=Predicting+Violent+Crimes+Per+Population;Machine+Learning+with+Ridge,+RandomForest,+XGBoost;Ensemble+Models+for+Improved+Accuracy" alt="Typing SVG" />
</p>

<!-- TECH BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
</p>

---

## 📌 Overview
This project applies **Machine Learning models** to predict **violent crimes per population** using socio-economic and demographic features.  
The pipeline includes **feature engineering, model training, and ensembling** for more robust predictions.  
Models used:
- **Ridge Regression**
- **Random Forest Regressor**
- **XGBoost**

---

## 📊 Dataset Information
| Feature                  | Description                                      |
|---------------------------|--------------------------------------------------|
| ID                        | Unique identifier for each sample                |
| PctPopUnderPov            | % of population under poverty                    |
| PctUnemployed             | % of unemployed individuals                      |
| PctBSorMore               | % of individuals with Bachelor's degree or more |
| PctNotHSGrad              | % of individuals without High School graduation |
| medIncome                 | Median income of population                      |
| ViolentCrimesPerPop (Target)| Violent crimes per 100K population (target)   |

- **Samples:** Provided in train.csv + test.csv  
- **Target:** `ViolentCrimesPerPop` (continuous value)

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/Rushorgir/CrimesPerPop
cd CrimesPerPop

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

---

## 🛠️ Feature Engineering

Key engineered features:
-	**EconomicVulnerability** = Poverty × Unemployment
-	**EducationAdvantage** = Higher Education ÷ Low Education
-	**IncomePovertyRatio** = log(Median Income) ÷ Poverty

Highly correlated raw features (e.g., numbUrban, medFamInc, etc.) were removed.

---

## 📈 Model Training & Ensemble
1.	**Ridge Regression** → With feature scaling.
2.	**Random Forest** → Hyperparameter tuned via RandomizedSearchCV.
3.	**XGBoost** → Hyperparameter tuned via RandomizedSearchCV.

**Finally, an ensemble prediction (average of all models) is generated for submission.**

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:FF4500&height=100&section=footer"/>
</p>
