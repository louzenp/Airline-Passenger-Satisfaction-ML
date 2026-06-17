#  Airline Passenger Satisfaction — ML Pipeline & XAI

[Русский](#-описание-проекта-русский) | [English](#-project-description-english)

---

##  Описание проекта (Русский)

Проект решает задачу **бинарной классификации удовлетворённости пассажиров авиакомпании** (`satisfied` / `neutral or dissatisfied`) на основе датасета [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) (~130 000 строк, 23 признака: демография, класс обслуживания, оценки сервисов на борту, задержки рейсов).

Работа выполнена в три последовательных шага (каждый — отдельный Jupyter Notebook), от честного baseline до интерпретируемой production-ready модели.

### Структура репозитория

| Файл | Содержание |
|---|---|
| `step3_pipeline_baseline.ipynb` | `sklearn.Pipeline` + `ColumnTransformer` без утечки данных; Logistic Regression (L1) как baseline; Stratified 5-Fold CV |
| `step4_ensembles_boosting.ipynb` | Random Forest и LightGBM с подбором гиперпараметров через `RandomizedSearchCV`; сравнение моделей по Accuracy / F1 / ROC-AUC / Precision / Recall |
| `step5_shap_xai_final.ipynb` | Интерпретация финальной модели через **SHAP** (TreeExplainer): global bar chart, beeswarm, dependence plots, waterfall для отдельных предсказаний |
| `train.csv` / `test.csv` | Обучающая и тестовая выборки |

### Методология

- **Pipeline без утечек данных** — вся предобработка (масштабирование числовых признаков, OneHot для категориальных, обработка сервисных шкал) обёрнута в `ColumnTransformer` и фитится только на train-фолде внутри кросс-валидации.
- **Baseline** — Logistic Regression с L1-регуляризацией (отбор признаков через обнуление коэффициентов).
- **Ансамбли** — Random Forest (бэггинг) и LightGBM (градиентный бустинг, leaf-wise рост деревьев), оба с `RandomizedSearchCV` (Stratified 5-Fold) по сетке ключевых гиперпараметров.
- **Объяснимость (XAI)** — SHAP TreeExplainer для победившей модели: какие признаки и почему влияют на предсказание, как на глобальном уровне, так и для отдельных пассажиров.

### Результаты

| Модель | Accuracy | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (L1, baseline) | 0.866 | 0.849 | 0.926 | 0.843 | 0.855 |
| Random Forest (tuned) | 0.964 | 0.958 | 0.995 | 0.973 | 0.944 |
| **LightGBM (tuned)** | **0.965** | **0.960** | **0.996** | 0.971 | 0.949 |

Лучшая модель — **LightGBM** после `RandomizedSearchCV`. Ключевые драйверы удовлетворённости по SHAP: **Online boarding**, **Inflight wifi service**, тип поездки (`Personal/Business Travel`) и **Inflight entertainment** — цифровой клиентский опыт оказался важнее традиционных параметров вроде еды на борту.

### Стек технологий

`Python`, `pandas`, `scikit-learn`, `LightGBM`, `SHAP`, `matplotlib` / `seaborn`, `Jupyter Notebook`

### Автор

Мурзабаев Нурсултан

---

##  Project Description (English)

This project tackles a **binary classification task for airline passenger satisfaction** (`satisfied` vs `neutral or dissatisfied`) using the [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset (~130,000 rows, 23 features covering demographics, travel class, in-flight service ratings, and flight delays).

The work is organized into three sequential steps (one Jupyter notebook each), progressing from a leak-free baseline to a tuned, interpretable production-ready model.

### Repository structure

| File | Content |
|---|---|
| `step3_pipeline_baseline.ipynb` | Leak-free `sklearn.Pipeline` + `ColumnTransformer`; Logistic Regression (L1) as baseline; Stratified 5-Fold CV |
| `step4_ensembles_boosting.ipynb` | Random Forest and LightGBM with hyperparameter tuning via `RandomizedSearchCV`; model comparison on Accuracy / F1 / ROC-AUC / Precision / Recall |
| `step5_shap_xai_final.ipynb` | Interpretation of the final model with **SHAP** (TreeExplainer): global bar chart, beeswarm plot, dependence plots, waterfall plots for individual predictions |
| `train.csv` / `test.csv` | Training and test datasets |

### Methodology

- **Leak-free pipeline** — all preprocessing (scaling numerical features, one-hot encoding categoricals, handling service-rating scales) is wrapped in a `ColumnTransformer` and fit only on the training fold within cross-validation.
- **Baseline** — L1-regularized Logistic Regression (built-in feature selection via coefficient shrinkage).
- **Ensembles** — Random Forest (bagging) and LightGBM (gradient boosting with leaf-wise tree growth), both tuned with `RandomizedSearchCV` (Stratified 5-Fold) over key hyperparameter ranges.
- **Explainability (XAI)** — SHAP TreeExplainer applied to the winning model to explain feature contributions both globally and for individual passenger predictions.

### Results

| Model | Accuracy | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (L1, baseline) | 0.866 | 0.849 | 0.926 | 0.843 | 0.855 |
| Random Forest (tuned) | 0.964 | 0.958 | 0.995 | 0.973 | 0.944 |
| **LightGBM (tuned)** | **0.965** | **0.960** | **0.996** | 0.971 | 0.949 |

The best-performing model is **LightGBM** after `RandomizedSearchCV` tuning. According to SHAP analysis, the top satisfaction drivers are **Online boarding**, **Inflight wifi service**, travel type (`Personal/Business Travel`), and **Inflight entertainment** — digital customer experience outweighs traditional factors like on-board food quality.

### Tech stack

`Python`, `pandas`, `scikit-learn`, `LightGBM`, `SHAP`, `matplotlib` / `seaborn`, `Jupyter Notebook`

### Author

Nursultan Murzabayev
