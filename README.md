## Predicting if it will rain the next day with clustering and supervised ML

*This project was completed as part of my cursus at Ironhack (a 9-week intensive coding bootcamp).*

The objective of this project was to **predict wether it would rain the next day**, with the underlying goal of practicing **classification tasks**, **clustering** and **hyper-parameter tuning**.

#### WHAT IS THE DATASET?

 - The dataset contains about **10 years of daily weather observations from numerous Australian weather stations**. The target `RainTomorrow` indicates wether it rained the next day or not (0: yes, 1: no). It can be found on [this Kaggle page](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). 
 - After dropping columns with high rates of null values, the dataset had the following structure:
	 - **142,193 days** x [ **16 weather indicators** + **whether it rained the next day or not** ]

#### DATA CLEANING

 - **Deal with outliers**: 4 features were winsorized:
	 - `Rainfall`
	 - `WindGustSpeed`
	 - `WindSpeed9am`
	 - `WindSpeed3pm`
 - **Deal with correlation**: 3 features were dropped:
	 - `MinTemp`
	 - `MaxTemp`
	 - `Pressure9am`
 - `Date` was transformed into `MeasureAge = 2020 - Date` to account for f global warming.

#### MODELING APPROACH & RESULTS

The primary objective of our model is to **reduce False Positives** (*the model predicts rain, but it is actually sunny, resulting in a loss of the harvest if the farmer doesn't water the field*), and **reduce False Negatives** as a secondary objective (*the model predicts a sunny weather, but it actually rains, resulting in a loss of water resources*).

Therefore, the *primary KPI* will be **Precision**, while the *secondary KPI* will be the **F1-Score**.

**STEP #1 - Reduce model complexity by creating a cluster on wind parameters using K-Means**

 - The Elbow method indicated that the optimal number of clusters created through the **K-Means algorithm** should be 4. Their characteristics are highlighted below:
	- *Cluster 0*: low wind speed
	- *Cluster 1* : increasing wind speed (mostly in the South direction)
	- *Cluster 2*: high wind speed (mostly in the West direction)
	- *Cluster 3*: decreasing wind speed
- The **Silhouette score** (0.21) and **Davies-Bouldin score** (1.66) highlight that the clusters are not highly differenciated.

Clusters 3D-plot:
![](https://raw.githubusercontent.com/gabrielleberanger/predicting-rain/master/graphs/clusters-3d-plot.png)

Mean of each wind feature by cluster (scaled from 0 to 1):
![](https://raw.githubusercontent.com/gabrielleberanger/predicting-rain/master/graphs/mean-of-each-wind-feature-by-cluster.png)

**STEP #2 -  Cross-validate models on the train set (10 folds) to identify the best performing one**
- This approach was tested with and without clustering, so that we can check if this prior grouping actually helped the model. In the end, **the clustering approach was not selected**.
-  **Best performer: CatBoostClassifier**
- Cross-validated model scores:
	- **Without clustering: Precision 76.6%, F1-Score 64.3%**
	- With clustering: Precision 75.7%, F1-Score 61.3%

**STEP #3 - Train the best performer (*CatBoostClassifier, without clustering*) on the whole train set, and measure its performance on the test set**
- Model scores on the test set: Precision 77.1%, F1-Score 75.9%

In parallel, we also tested **hyper-parameter tuning on the RandomForestClassifier model**, which was also one of the best performers during the cross-validation phase. The selected model achieved a 79.1% Precision score, with the following parameters: `{'n_estimators': 450, 'max_features': 'log2', 'criterion': 'entropy', 'class_weight': 'balanced'}`
