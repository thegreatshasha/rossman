import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from helpers import rmspe, mean_error, split_data
from sklearn.cross_validation import train_test_split

def train(X, Y):
    rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)
    rf.fit(X.as_matrix(), Y.as_matrix())
    return rf

def test(model, X, Y):
    output = model.predict(X.as_matrix())
    mean_score = mean_error(Y, output)
    rmspe_score = rmspe(Y.as_matrix(), output)
    return mean_score, rmspe_score

def submit(model, X):
    x = X.drop('Id', axis=1).as_matrix()
    output = model.predict(x)
    
    result = pd.DataFrame(X.Id).join(pd.DataFrame(output,columns=['Sales']))
    result.sort('Id').to_csv('submission.csv')

def main():
    training_input_file = 'data/training_vector.csv'
    submit_input_file = 'data/test_vector.csv'

    data = pd.read_csv(training_input_file)
    submit_data = pd.read_csv(submit_input_file)

    training_data, test_data = train_test_split(data, test_size=0.5)
    
    model = train(training_data.drop(['Sales'], axis=1), training_data['Sales'])
    mean_score, rmspe_score = test(model, training_data.drop(['Sales'], axis=1), training_data['Sales'])

    print "Mean Score: %f, Rmspe Score: %f" % (mean_score, rmspe_score)

if __name__ == "__main__":
    main()