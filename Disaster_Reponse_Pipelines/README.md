# Disaster Response Pipelines with Figure Eight

## About :

In this project, I have applied my data engineering and NLP skills to 
analyze disaster data from Figure Eight to build a model for an API that 
classifies disaster messages. I learnt how to create an ETL and a machine 
learning pipeline to categorize messages sent during disaster events. These 
maybe then directed to, according to the classified category, the appropriate
disaster relief agency. The html templates were already provided by udacity 
and the visualizations used have been coded by me.

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

All the below folders are contained in disaster_response_pipeline_project.

1. app: contains all the files relating to the web API development. 
2. data: contains the raw data, data processing script and its 
corresponding jupyter notebook.
3. models: contains the model training and saving script along with
 its jupyter notebook.
4. imgs: contains screenshots of tha app.

'Note: Look into each folder's README to find more information about 
its contents.'

## Requirements :

1. Flask
2. NLTK
3. numpy
4. plotly
5. pandas
6. sklearn
7. sqlalchemy  

## Screenshots :

Here are some screeshots of the API:

![yay](https://github.com/beatboxerish/DSND/Disaster_Response_Pipelines/disaster_response_pipeline_project/imgs/'Screenshot from 2019-03-12 02-19-30.png')

