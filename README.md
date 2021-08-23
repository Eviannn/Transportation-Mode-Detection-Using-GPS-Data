# Transportation-Mode-Detection-Using-GPS-Data
Final Year Project in SEU, 2021.06


# Abstract of Project

Getting the information of residents’ travel needs is a prerequisite for guiding the construction of transportation infrastructure and urban transportation planning. In the context of the era of big traffic data, this paper designs a travel demand information extraction algorithm, based on Global Positioning System (GPS) trajectory data, extracts motion feature indicators after preprocessing, conducts exploratory data analysis to discover transportation modes and finds that there are obvious differences in the distribution characteristics of speed and acceleration, and these differences can be used to identify different transportation modes. On this basis, the travel characteristics are selected in combination with the Pearson correlation coefficient matrix and the Random Forest model, and finally the maximum speed, speed standard deviation, non-zero average speed, maximum acceleration, acceleration standard deviation and average acceleration corresponding to non-zero speed are used as characteristic variables. In the travel demand information algorithm design, the Support Vector Machine algorithm and the XGBoost algorithm are used and compared, and the latter is found to be better to identify different transportation modes. In order to further improve the recognition accuracy, this paper uses the Stacking model fusion method, combined with a variety of machine learning algorithms to extract travel demand information, and finally the weighted macro average F1-score on the test set reaches 89.43%. Through the sensitivity analysis of data sampling frequency, it is found that selecting the lowest data sampling frequency appropriately can improve the efficiency of data storage while ensuring the recognition effect of the algorithm. Through the travel demand information extraction algorithm designed in this paper, the travel demand information of residents can be grasped in a timely manner to meet the needs of the rapid development of contemporary urban transportation.

KEY WORDS: GPS trajectory data, transportation mode, travel characteristics, machine learning algorithm


# Data Source
https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fgeolife%2Fdefault.aspx


# Files Including
### Original Data Format
1. Example Data
### Transportation Mode Feature Extract
2. Data Matching.py
3. Data Process.py
4. Feature Calculate.py
5. Feature Extract.py
### Feature Data Analysis
6. Data Analysis.py
7. Data Analysis.html
### Using RF Model to Select Feature
8. RF_feature select.py
### Model Establish and Result
9. SVM.py
10. Learning Curve of SVM.png
11. XGBoost.py
12. Learning Curve of XGBoost.png
13. Stacking Model.py
14. Learning Curve of Stacking.png
