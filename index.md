<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Selected projects in data science, machine learning and physics

Here I present projects (commercial or research) I have done in a form of short case studies - a glimpse of my professional experience.

<br><br>

## Origination Analytics

### Background

Origination of commodities directly from farmers is an essential component of value chain management for large agribusiness companies. Understanding which farmers fall within our right-to-win area is crucial for prioritization and pricing decisions. In this project, we developed a tool that integrates government agriculture data, internal sales data, and publicly available information to visualize the competitive landscape for origination managers.

### Methods

Initially, the project began as a modernization of legacy code stored in Jupyter notebooks, which required manual execution by a data scientist upon business request. Our initiative transformed it into a self-service application for business users. The UI was built with Streamlit, enabling users to browse and schedule simulations for various locations with business-specific parameters, as well as explore visualizations of essential geospatial data. The computations are deployed as jobs in GCP Cloud Run, with results stored in BigQuery.

### Results

The tool significantly enhanced the day-to-day operations of origination managers, enabling them to focus their efforts on the most promising deals and strengthen relationships with key farmers.

### Tools

- Python - Streamlit, NumPy, Pandas
- GCP - BigQuery, Dataform, Cloud Run
- Infrastructure - Terraform

## Contract Booking Prediction

### Background

This project was part of a larger initiative for a leading ABCD agribusiness company specializing in food solutions for industrial customers. The objective was to enhance sales team productivity by enabling them to prioritize high-value negotiations. Specifically, we developed predictive models to estimate the probability of contract closure within a specified timeframe, leveraging historical customer interaction data to guide strategic decision-making.

### Methods

The project presented significant data quality challenges that precluded straightforward modeling approaches. The dataset exhibited severe class imbalance, with only 20% of contracts being successfully booked within the target period. To address this, I employed SMOTE (Synthetic Minority Over-sampling Technique) to rebalance the training data. I then developed and compared multiple classification models—including Random Forest, XGBoost, and Support Vector Machines—to maximize prediction accuracy. Model performance was evaluated using precision, recall, and F1-score metrics to ensure reliable probability estimates for business use.

### Results

The final ensemble achieved a minimum accuracy of 0.85 across all contract categories, enabling the sales team to effectively prioritize negotiations with the highest conversion potential. This data-driven approach contributed to multiple high-margin contract acquisitions, demonstrably improving the company's win rate and revenue outcomes. The system was deployed with automated email notifications to alert sales managers of high-probability opportunities.

### Tools

- Python - NumPy, Pandas, scikit-learn, XGBoost
- SQL
- Data Science - Imbalanced learning (SMOTE), classification algorithms
- Deployment - FastAPI, Docker, SMTP server

## Detecting Gear Fault with Deep Learning-based Methods

### Background

Toothed gears are critical components in mechanical systems, and their failure can lead to costly downtime and safety hazards. Pitting failure, characterized by surface fatigue and material removal, is one of the most common failure modes in gear systems. Traditional detection methods often lack the sensitivity to identify early-stage damage. This project aimed to develop a robust, automated method for detecting pitting failures in toothed gears by analyzing vibrational signals from the gear case using deep learning techniques.

### Methods

The approach utilized an autoencoder deep neural network architecture trained in a semi-supervised manner on vibrational signal data collected from a power circulation test stand. The autoencoder reconstructed gear case vibrational signals, learning to represent normal gear behavior in a compressed latent space. For classification, I implemented latent data convex hull-based clustering to distinguish between healthy and damaged gears. The method was validated against traditional techniques including Principal Component Analysis (PCA) and Generative Adversarial Networks (GANs) to benchmark its performance and generalization capabilities.

<img src="images/gear-ae.png?raw=true" />

*Figure: Autoencoder architecture for gear fault detection. Source: M. Batsch & B. Kiczek, Appl. Sci. 2024, 14(12), 5282*

### Results

The proposed method achieved exceptional performance with an F1-measure of 0.99, including 100% accuracy in failure detection and 98.9% accuracy in normal state prediction. Notably, the system demonstrated high sensitivity, successfully detecting even slight surface damage indicative of initial pitting. The deep learning approach significantly outperformed linear techniques like PCA and showed superior generalization compared to nonlinear methods such as GANs. These results were published in a peer-reviewed journal and have practical implications for predictive maintenance in industrial applications.

### Tools

- Python - NumPy, Pandas, scikit-learn, Keras, TensorFlow
- Signal Processing - Vibration analysis, Fourier transform
- Machine Learning - Autoencoders, semi-supervised learning, clustering algorithms


## Ultra-Wide Band Localization

### Background

Indoor localization systems are essential components of modern technology, enabling the tracking of people, vehicles, and assets in environments where GPS signals are unavailable. Each deployment requires careful consideration of client needs and building-specific constraints. For this project, a medium-sized industrial client required a solution to track forklifts and assets in a warehouse environment. We proposed an Ultra-Wide Band (UWB) system with transmitters and anchors, and I developed algorithms to localize transmitters based on distance measurements from strategically placed anchors throughout the facility.

### Methods

Real-world radio signal data presents inherent challenges, as various environmental factors can interfere with distance measurements. The solution required robustness against small to medium signal disturbances. I formulated the problem using residual distances between measured (noisy) and actual distances:

$$
u_{(i)}(\vec{x}_T) = (x_{a (i)} - x_{T})^2 + (y_{a (i)} - y_{T})^2 + (z_{a (i)} - z_{T})^2 - d_{(i)} ^2
$$

The objective function aggregates these residuals across all anchor-transmitter pairs:

$$
F(\vec{x}_T) = \sum_{i}^{N_a}{u_{(i)}(\vec{x}_T)^2}
$$

By squaring the residuals, we ensure a positive objective function where the transmitter's position corresponds to the global minimum. To address the challenge of finding this minimum in noisy conditions, I implemented DBSCAN clustering for outlier detection, ensuring robust and accurate position estimates.

### Results

The deployed solution achieved precise localization with a deviation of less than 25 cm, enabling reliable real-time tracking of warehouse assets and significantly improving operational efficiency.

### Tools

- Python - NumPy, SciPy, scikit-learn
- SQL
- Optimization - Nonlinear least squares, DBSCAN clustering
- Deployment - FastAPI, Docker

<br><br>

## Bluetooth LE Beacon Localization

### Background

This project involved developing an indoor navigation system for medical facilities, including hospitals and clinics. The objective was to enable patients and visitors to navigate efficiently to their destinations, such as doctor's offices or patient rooms, improving the overall facility experience and reducing staff burden from providing directions.

### Methods

Given deployed Bluetooth Low Energy (BLE) beacons throughout the facility and collected signal data, I developed machine learning solutions to determine user positions with maximum precision. Initially, I investigated RSSI (Received Signal Strength Indicator) patterns to train distance estimation models. However, BLE technology exhibits significant signal instability, necessitating extensive data preprocessing and feature engineering. I then developed and compared regression models using XGBoost and Support Vector Machines (SVM) to predict user positions within the facility's coordinate system, optimizing for both accuracy and real-time performance.

### Results

The final solution achieved substantial improvements in position accuracy, with a mean localization error of less than 2 meters—exceptional performance for BLE-based systems. This precision enabled reliable navigation assistance and opened possibilities for additional location-based services within medical facilities.

<img src="images/BTLE-XGboost.png?raw=true" />

### Tools

- Python - NumPy, Pandas, scikit-learn, XGBoost
- R - tidyverse
- Machine Learning - Regression algorithms, signal processing

<br><br>

## Radio Tomographic Imaging Reconstruction

### Background

Unlike traditional indoor localization systems that require users to carry active devices, Radio Tomographic Imaging (RTI) enables passive detection of individuals without any carried electronics. The system operates by encircling a monitored area with radio sensors that continuously exchange data packets and measure received signal strength. Human presence can be detected through the absorption and scattering of electromagnetic waves by the body, making this approach ideal for privacy-conscious applications such as occupancy monitoring in medical facility waiting areas.

### Methods

Traditional RTI approaches rely on sensitivity matrices determined by room geometry and sensor distribution, requiring numerous mathematical approximations that compromise accuracy. I developed a machine learning-based approach using deep neural networks in TensorFlow, eliminating the need for these approximations. The neural network architecture learned complex relationships between signal patterns and spatial occupancy, significantly improving reconstruction quality without requiring computationally expensive pseudo-inverse calculations. This approach also provided greater robustness to sensor placement variations and environmental changes.

### Results

The neural network-based solution dramatically improved image reconstruction quality, enabling clear identification of individual persons in monitored spaces. This capability allowed for accurate occupancy counting and monitoring in clinic waiting areas, supporting both capacity management and safety protocols. The system successfully distinguished multiple individuals simultaneously, as shown in the visualization below. These findings were published in the peer-reviewed journal <a href="https://www.mdpi.com/1996-1073/16/1/275">Energies</a>, contributing to the academic understanding of RTI applications.

<img src="images/RTI-nn.png?raw=true" />

### Tools

- Python - NumPy, Matplotlib, scikit-learn, TensorFlow
- Apache Kafka
- Deep Learning - Convolutional neural networks, image reconstruction
- Deployment - FastAPI, Docker, Azure App Service

<br><br>

## Medical Disease Risk Prediction

### Background

This project was part of a comprehensive initiative to modernize a medical facility management system, including patient documentation and clinical decision support capabilities. The system aimed to assist physicians by providing data-driven risk assessments for three prevalent conditions: obesity, coronary heart disease, and diabetes. A dataset comprising patient medical features—including blood pressure, cholesterol levels, weight, height, and age—was annotated by a panel of medical doctors with estimated disease probabilities. The dual objective was to explore how machine learning models could accurately predict disease risk while providing interpretable insights into key contributing factors for clinical use.

### Methods

The dataset contained over 10,000 patient records but exhibited substantial missing data, requiring careful imputation using k-Nearest Neighbors (kNN) methods to preserve data relationships. I developed three separate XGBoost regression models tailored to each condition's complexity—while obesity prediction required only linear relationships, coronary heart disease and diabetes benefited from more sophisticated nonlinear modeling. Each model was rigorously validated and optimized to achieve R² scores exceeding 0.975, ensuring reliable probability estimates. The inherent explainability of XGBoost enabled comprehensive feature importance analysis, revealing that age, cholesterol levels, and blood pressure were the dominant predictive factors—aligning with established medical knowledge and enhancing physician confidence in the system.

### Results

The models achieved exceptional predictive performance across all three conditions, demonstrating the viability of machine learning for clinical decision support. These preliminary results proved instrumental in securing additional funding from stakeholders and international headquarters, enabling the development of a full-scale AI-powered diagnostic assistance platform for medical facilities. The project successfully bridged data science capabilities with clinical requirements, establishing a foundation for AI integration in healthcare workflows.

### Tools

- Python - NumPy, scikit-learn, XGBoost, Pandas
- R - tidyverse, UBL (Unbalanced Learning)
- Machine Learning - Regression, feature importance analysis, missing data imputation