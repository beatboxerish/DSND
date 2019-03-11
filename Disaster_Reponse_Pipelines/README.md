# Disaster Response Pipelines with Figure Eight

## About :

In this project, I have applied my data engineering and some NLP skills to 
analyze disaster data from Figure Eight to build a model for an API that 
classifies disaster messages. I learnt how to create an ETL and a machine 
learning pipeline to categorize messages sent during disaster events. These 
maybe then directed to, according to the classified category, the appropriate
disaster relief agency.

## Usage:

1. Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in sql database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* To run ML pipeline that trains classifier and saves the model:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:
    `python run.py`

3. Go to http://0.0.0.0:3001/ to check out the API.

## Files/Folders :

1. 
