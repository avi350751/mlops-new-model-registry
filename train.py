import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/avi350751/my-datasets/main/pima_diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create the RF model
rf = RandomForestClassifier(random_state=42)

# Defining the parameters for 
param_grid = {
		'n_estimators': [5,10,30,50,100],
		'max_depth' : [None, 10,20,30,40]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid= param_grid, cv=5, n_jobs=-1, verbose=3)

mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run(run_name='avi-grid-search-exp1', description = 'Best hyperparameter trained on RF model') as parent:
	grid_search.fit(X_train, y_train)

	# log all the children runs
	for i in range(len(grid_search.cv_results_['params'])):
		
		#print(i)
		with mlflow.start_run(nested=True) as child:
			mlflow.log_params(grid_search.cv_results_['params'][i])
			mlflow.log_metric('accuracy', grid_search.cv_results_['mean_test_score'][i])
	
	
	# Displaying best parameters and best score
	best_params = grid_search.best_params_
	best_score = grid_search.best_score_

	# log 
	mlflow.log_params(best_params)
	mlflow.log_metric('accuracy', best_score)
	
# log data
	train_df = X_train
	train_df['Outcome'] = y_train
	
	train_df = mlflow.data.from_pandas(train_df)
	mlflow.log_input(train_df, 'train')

	test_df = X_test
	test_df['Outcome'] = y_test
	
	test_df = mlflow.data.from_pandas(test_df)
	mlflow.log_input(test_df,'validation')

# source code
	mlflow.log_artifact(__file__)

# model
	#signature = mlflow.models.infer_signature(X_train, grid_search.best_estimator_.predict(X_))
	mlflow.sklearn.log_model(grid_search.best_estimator_,'random_forest')
# tags
	mlflow.set_tag('author','avi')
	mlflow.set_tag('model','random_forest')

	print('best parameter: ',best_params)
	print('best score : ',best_score)