import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train(X, Y):
    X = X.as_matrix()
    Y = Y.as_matrix()
    rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)
    rf.fit(X, Y)
    import pdb; pdb.set_trace()

def main():
    input_file = 'data/training_vector.csv'

    training_data = pd.read_csv(input_file)

    train(training_data.drop(['Sales'], axis=1), training_data['Sales'])

if __name__ == "__main__":
    main()