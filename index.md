<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Selected projects in data science, machine learning and physics

Hereby I present projects (commercial or research) I have done in a form of short case studies - a glimpse of my professional experience.

<br><br>

## Ultra-wide band localisation

### Background

Indoor localisation systems are an important part of modern technology, as they allow to track people, vehicles, assets and many more inside buildings, where GPS signal is unavailable. There is no a single solution on the market, as each deployment is different and must be consulted with client’s needs and the building in question iteslf.

In this case, a medium-sized industrial client sought a solution which would allow tracking of forklifts and assets in a warehouse. The transmitters and anchors using UWB were proposed for this solution. My duty was to prepare algorithms which localise the transmitter given distances between it and a set of anchors placed in the building.

### Methods

Dealing with real-world data always has an element of trickery. Different reasons can influence the radio signal from the device and hence disturb the distance measurement. Therefore the used method had to be robust against small to medium disturbances.

For this reason I considered residual distances computed by subtracting measured (noisy) distances from real distances

$$
u_{(i)}(\vec{x}_T) = (x_{a (i)} - x_{T})^2 + (y_{a (i)} - y_{T})^2 + (z_{a (i)} - z_{T})^2 - d_{(i)} ^2
$$

Naturally we do not know what are the real distances, but in ideal scenario the above residuals should be zero. We form and objective function from these residuals, by adding them for each anchor-transmitter pair

$$
F(\vec{x}_T) = \sum_{i}^{N_a}{u_{(i)}(\vec{x}_T)^2}
$$

by squaring $u$ we are sure that the objective function is positive. Such function shall be minimised - the position of the transmitter is the place where the function is minimal.

However finding the global minimum is usually tricky for optimization algorithms. For this reason I used unsupervised machine learning DBSCAN for outlier detection, to make sure that computed point is accurate.

### Results

The deployment of this algorithmic solution allowed for precise localisation (less than 25cm deviation) of the users of the system.

### Tools

- Python - numpy, scipy, scikit-learn
- SQL
- Mathematics
- Deployment - FastAPI, Docker

<br><br>

## Bluetooth LE beacon localisation

### Background

In this project I was a part of a team developing an indoor localisation system for medical facilities, such as hospitals or clinics. The goal was to allow a user to navigate inside a building to a desired destination, usually a doctor’s office or patient’s room.

### Methods

Given a set of hardware inside a building and a dataset, my duty was to create a machine learning solution, which could localise the user as precisely as possible.

In the early stage of project I investigated the RSSI of Bluetooth (BT) signal in order to train a model, which could reproduce the distance between the user and each beacon. The task itself proved to be tricky, as BT is quite an unstable technology and some intensive data preparation was needed.
Alternatively I created regression models (XGBoost and SVM), which were able to predict the position of the user in a given reference frame.

### Results

I achieved a huge improvement in the position designation. The mean distance between real and predicted point was less than 2m which is really tiny for this technology. Such precision allowed for a reliable determination of user’s position.

<img src="images/BTLE-XGboost.png?raw=true" />

### Tools

- Python - numpy, pandas, scikit-learn, xgboost
- R - tidyverse

<br><br>

## Real estate dashboard

### Background

This little side project combines web scrapping with interactive data visualisation. I made it entirely for myself for analysis of real estate price trends in my hometown. I used OLX, which is a popular marketplace website in Poland, as a data source. The collected and processed data were saved to csv files, which were interpreted by a plotly dashboard app.

Plotly has a capability to create various dashboards for easy, from the creator’s side, presentation and effective data visualisation.

### Methods

Web sniffer is built with Beautiful Soup 4 framework in Python. It browses most populat sites from “flats for sale” query, and on each site it extracts URLs to every advertisement, which are saved on a list. Then another function checks each found advertisment URL and looks for valuable information, such as size of the flat, its price, age of a building etc.

These data are collected in a tabular form and finally stored as a csv file. To prevent the sniffer from being banned by the website it takes random breaks from time to time.

As I have already mention the dashboard was created using plotly. For visualisation I created a possibility to explore each time snapshot, which can be chosen from the drop-down menu on the top of the dashboard.

On the other hand I plan to add a subpage where temporal analysis of some quantities is possible.

### Results

By finalising this hobby project I provided myself with a nicely presented summary of local real estate market.

<img src="images/RealEstateDashboard.png?raw=true" />

### Tools

- Python (Beautiful soup, numpy, pandas, plotly dash)

<br><br>

## RTI Reconstruction

### Background

Two previously mentioned indoor localisation techniques I worked on have a common feature - they require an active device at the user’s position. However, is it possible to localise a person that does not have any electronics with them? Yes, and we can use Radio Tomographic Imaging (RTI) for this.

A room in question is then encircled with an assembly of radio sensors. Simply speaking, these sensors exchange packages of data and measure the received signal strength. This information can be used for determining whether a person (whose body absorbs a bit of these non-invasive electromagnetic waves) is inside.

### Methods

The standard methodology utilises a so-called sensitivity matrix, which is determined by the room geometry and sensor distribution. However, computing such requires lots of mathematical approximations, hence leads to pretty inaccurate results.

The usage of machine learning brings lots of possibilities to the problem. Mainly, because it does not require us to make any compromises in the pseudo inverse calculation. Additionally, incredible powers of neural networks to learn complex relations in the data gives us very accurate results.

In this project I created multiple neural networks using tensorflow. These models proved to be extremely useful in RTI reconstruction.

### Results

The usage of neural networks incredibly increased the quality of assessed images. Every person could be separately seen in the reconstructions, therefore it allowed for calculation and control of number of people in the waiting area of a clinic.

<img src="images/RTI-nn.png?raw=true" />

In the image above we can see a person in the room.
The results were also published in <a href="https://www.mdpi.com/1996-1073/16/1/275">Energies</a> journal.

### Tools

- Python - numpy, matplotlib, scikit-learn, tensorflow
- Apache Kafka
- Deployment - FastAPI, Docker, Azure App Service

<br><br>

## Medical project - disease prediction

### Background

This assignment was a part of a bigger project, a sketch of an update for an application for medical facility management, as well as creating and managing patient’s documentation. I received a dataset containing set of medical features of patients, such as blood pressure, cholesterol levels, weight, height, age etc.

The documentation of every patient was carefully reviewed by a group of medical doctors. Given that, they estimated the probabilities of three diseases. These were obesity, coronary heart disease and diabetes.

The goal of this project was twofold. First, a scientific curiosity - how a machine learning model can model the probability of disease occurence. Additionaly, it is interesting which features are the most important for the particular result. Second, the usage of this models in the production environment to help physicians during the diagnosis process.

### Methods

The dataset was relatively big (more than ten thousand records), but it had lots of missing values. It required modelling of this NA’s, for this I used kNN methods. Next I prepared three regression models based on XGBoost. While obesity diagnosis was simple enough for a linear model, remaining two diseases required more complex solution. Finally I managed to train models and acquire $R^2 > 0.975$ for each regressor.

The explainability of XGBoost also allowed for feature importance analysis. It turned out that the most important features were age, cholesterol levels and blood pressure. This is not surprising, as these are the most common factors in the diagnosis process.

### Results

These preliminary results were important to push the project further and gain additional financing from the stakeholders and CEO of the foreign headquarters. Funding acqusition for AI-based expert engine for medical facilities was made possible.

### Tools

- Python - numpy, scikit-learn, xgboost, pandas
- R - tidyverse, UBL (for data augumentation)

