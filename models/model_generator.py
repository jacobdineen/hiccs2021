import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

from utils.Dataloader import Dataloader, check_existence


def create_model_dirs(sample_size, verbose=False):
    dirName = os.getcwd() + '\\models\\{}'.format(sample_size)
    try:
        os.mkdir(dirName)
        if verbose:
            print("Directory ", dirName, " Created ")
    except FileExistsError:
        pass


def save_models(sample_size, model, model_name, verbose=False):
    create_model_dirs(sample_size)
    file_name = os.getcwd() + '\\models\\{}\\{}.pkl'.format(
        sample_size, model_name)
    pickle.dump(model, open(file_name, 'wb'))
    if verbose:
        print(f'Model saved @ {file_name}')


class Gridsearch:
    '''
    Class to perform Gridsearch w\\ 4 specified estimators

    Parameters
        dataloader: object, pass in to avoid reloading the full dataset into memory each time
        kfold: int, number of folds for CV
        RFC: obj, Random Forest Classifier
        GBC: obj, Gradient Boosting Classifier
        LR: obj, Logistic Regression Classifier
        ANN: obj, sklearn neural network Classifier
    Returns
        Stores serialized models in respective folders given sample sizes

    Example:
        dataloader = Dataloader(file_location='data\\loan.csv', destination='data\\loan_cleaned.csv')
        gs = Gridsearch(dataloader = dataloader)


        size = [20000,40000,80000,160000]
        for i in size:
            gs.generate_models(sample_size= i)



    '''
    def __init__(self, dataloader):

        self.dataloader = dataloader

        # self.model_names = ['RFC', 'GBC', 'LR', 'NN']
        self.model_names = ['RFC', 'GBC', 'LR']
        self.kfold = 5

        self.RFC = RandomForestClassifier()
        self.GBC = GradientBoostingClassifier()
        self.LR = LogisticRegression()
        self.ANN = MLPClassifier()
        self.objects = {}
        self.model_paths = {}

        self.param_grid = {
            self.RFC: {
                "n_estimators": [50],
                "criterion": ['gini', 'entropy'],
                "max_features": ['auto', 'sqrt', 'log2']
            },
            self.GBC: {
                'n_estimators': [1000, 2000, 5000],
                "criterion": ['friedman_mse', 'mse'],
                "max_features": ['auto', 'sqrt', 'log2']
            },
            self.LR: {
                'penalty': ['l1', 'l2'],
                "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }
        }
        

    def generate_models(self, sample_size, overwrite = False):
        '''
        Parms
            sample_size: int, specifies how many total samples we want. 
                        We use datalaoder methods to enforce full class balance before train\\test splits
        '''
        print('-' * 50)
        print('Sample Size = ', sample_size)
        print('-' * 50)

        X_train, X_test, y_train, y_test, encoder, unrolled_features = self.dataloader.fetch_data(num_samples = sample_size,   train_size=0.8)
        self.objects[sample_size] = (X_train, X_test, y_train, y_test, encoder, unrolled_features) # We need to store all objects for later use

        counter = 0
        #Grid Search, store best estimator & write to disk.
        model_paths = []
        for k, v in self.param_grid.items():
            file_name =  '.\\models\\{}\\{}.pkl'.format(sample_size, self.model_names[counter])
            model_paths.append((self.model_names[counter], file_name))
            if not check_existence(file_name) and not overwrite:
            #         print('Constructing & Grid Searching for Estimator: {}'.format(model_names[counter]))
                temp_model = GridSearchCV(k,
                                        param_grid=v,
                                        cv=self.kfold,
                                        scoring="accuracy",
                                        n_jobs=12,
                                        verbose=2)
                temp_model.fit(X_train, y_train)

                print(
                    f"{self.model_names[counter]} Best Score: {(temp_model.best_score_ * 100):.2f}%"
                )


                save_models(sample_size, temp_model.best_estimator_,
                            self.model_names[counter])

                counter += 1
            else:
                print(f'model already exists {file_name}')
                counter += 1
        self.model_paths[sample_size] = model_paths
    




