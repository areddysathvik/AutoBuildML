# # AutoBuildML

AutoBuildML is a personal project that automatically builds machine learning models, taking care of the entire ML flow and providing an interactive experience.


<img src="https://img.freepik.com/premium-photo/robot-hand-making-contact-with-human-hand-cyborg-hand-finger-pointing-technology-ai_889761-2093.jpg">

## Steps Performed by the Pipeline

### Step 1: Data Preprocessing
- Handling null values
- Removing duplicates
- Scaling numerical features
- Encoding categorical features

### Step 2: Feature Selection
- Using ANOVA (Analysis of Variance) to select the best features

### Step 3: Model Building

#### For Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)

#### For Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

### step 4: you can also use your custom record for predictions

## Dockerized
- The application is Dockerized for easy deployment.

## How to Use

You can pull the Docker image of this application from Docker Hub:

    docker pull areddisathvik/autobuildml:1.0


Usage

To run the AutoBuildML application, follow these steps:

Clone this repository.
Navigate to the project directory.
Build the Docker image:

    docker build -t autobuildml:1.0 .

Access the application in your web browser at
        
    http://localhost:8501.

Enjoy building machine learning models effortlessly with AutoBuildML!
