import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train(X, Y):
    X = X.as_matrix()
    Y = Y.as_matrix()
    rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)
    rf.fit(X, Y)
    score = (abs(rf.predict(X) - Y)).mean()
    return rf, score

def test(model, X):
    x = X.drop('Id', axis=1).as_matrix()
    output = model.predict(x)
    
    result = pd.DataFrame(X.Id).join(pd.DataFrame(output,columns=['Sales']))
    result.sort('Id').to_csv('submission.csv')

def main():
    training_input_file = 'data/training_vector.csv'
    test_input_file = 'data/test_vector.csv'

    training_data = pd.read_csv(training_input_file)
    test_data = pd.read_csv(test_input_file)

    model, score = train(training_data.drop(['Sales'], axis=1), training_data['Sales'])
    print "Score is: %f\n" % score
    test(model, test_data)

if __name__ == "__main__":
    main()