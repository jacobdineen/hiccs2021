import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def check_existence(file):
    if os.path.isfile(file):
        # print('-' * 50)
        # print("cleaned file {} exists".format(file))
        # print('-' * 50)
        return True
    else:
        # print('-' * 50)
        # print(
        #     "file {} does not exist. Download from https:\\\\www.kaggle.com\\wendykan\\lending-club-loan-data#loan.csv"
        #     .format(file))
        # print('-' * 50)
        return False


def get_cols_of_interest(file='./data/columnsOfInterest.txt'):
    with open(file, 'r') as filehandle:
        columns = json.load(filehandle)
    return columns



class Dataloader:
    '''
    Class to load\\transform\\encode data as needed for SHAP. 
    Fetches the first num_samples \\ 2 observations from each class before running through preprocessing

    Parameters
        file_location: str, path to lending club original file
        destination: str, path for storage of cleaned data
    
    Returns
        OHE of categorical feature expansion, all feature names, original dataframe, train test splits in encoded format

    '''

    def __init__(self, file_location, destination):
        self.file = file_location
        self.destination = destination
        self.response = ['Charged Off', 'Fully Paid',]  # filter on response type
        self.columns = get_cols_of_interest()

    def _get_feature_names(self):
        '''
        Store Original Column\\Feature Mapping for lookups
        '''
        self.feature_names = self.columns
        self.continuous_features = self.df.iloc[:, :-1].select_dtypes(
            include=[np.float64, np.int64]).columns
        self.categorical_features = self.df.iloc[:, :-1].select_dtypes(
            include=[np.object]).columns
        self.categorical_features_index = [
            self.df.columns.get_loc(c) for c in self.categorical_features
            if c in self.df
        ]

    def _transform_y(self):
        '''
        Binary encoder for the response var. Likely 0 = Fully Paid, 1 = Charged Off. Need to check
        '''
        self.y = np.array(self.df)[:, -1]  # response var
        le = LabelEncoder()  # instantiate label encoder
        le.fit(self.y)  # fit label encoder to response
        self.y = le.transform(self.y)  # encode response

    def _transform_x(self):
        self.x = np.array(self.df)[:, :-1]  # Get all Independent Vars

        self.categorical_names = {}
        #Try-catch because something is breaking on one of the features
        try:
            for feature in self.categorical_features_index:
                le = LabelEncoder()  # instantiate label encoder
                le.fit(self.x[:, feature]
                       )  # fit label encoder to categorical features
                # transform categorical features
                self.x[:, feature] = le.transform(self.x[:, feature])
                # get categorical feature class names and append to dict
                self.categorical_names[feature] = le.classes_
        except:
            pass

        # all features as a list. Unwrapping encoding
        self.cat_features_unrolled = []

        for k in self.categorical_names.values():
            for i in k:
                self.cat_features_unrolled.append(i)

        # Make Features dicepherable
        self.cat_features_unrolled[:2] = [
            'Term:' + i for i in self.cat_features_unrolled[:2]
        ]
        self.cat_features_unrolled[2:9] = [
            'Loan_Grade:' + i for i in self.cat_features_unrolled[2:9]
        ]
        self.cat_features_unrolled[9:44] = [
            'Loan_SubGrade:' + i for i in self.cat_features_unrolled[9:44]
        ]
        self.cat_features_unrolled[44:56] = [
            'Employment_Length:' + i for i in self.cat_features_unrolled[44:56]
        ]
        self.cat_features_unrolled[56:62] = [
            'Home_Ownership:' + i for i in self.cat_features_unrolled[56:62]
        ]


        # Concatenate encoded features + continuous features
        self.unrolled_features = self.cat_features_unrolled + list(
            self.continuous_features)

    def _load_data(self, num_samples, random = False):
        '''
        use random = True if exploring with samples of same size, else, use False for variable sample size
        '''
        if check_existence(self.destination):
            iter_csv = pd.read_csv(self.file,
                                   iterator=True,
                                   chunksize=10000,
                                   low_memory=False,
                                   index_col=False)  # batch load into memory
            self.df = pd.concat([
                chunk[chunk.loan_status.isin(self.response)] for chunk in iter_csv
            ])  #concat chunks of file

            #Partition into equally sized sets and concatenate
            split = int(num_samples / 2)
            
            #Used for different same-sized samples
            if random:
                self.pos = self.df[self.df['loan_status'] == 'Fully Paid'].sample(split)
                self.neg = self.df[self.df['loan_status'] == 'Charged Off'].sample(split)
            else:
                self.pos = self.df[self.df['loan_status'] == 'Fully Paid'][:split]
                self.neg = self.df[self.df['loan_status'] == 'Charged Off'][:split]
                
            self.df = pd.concat([self.pos, self.neg])
            self.df = self.df[self.columns]  #subset on cols
        

    def _clean_data(self, num_samples, random = False):
        self._load_data(num_samples, random = random)
        self.df['emp_length'].fillna(value='unknown',
                                     inplace=True)  #error handling on NAN
        self.df.fillna(value=0, inplace=True) 
        self._get_feature_names()
        self._transform_x()
        self._transform_y()
        print('data transformation complete')

    def fetch_data(self, num_samples, train_size, random = False):
        self._clean_data(num_samples)

        #tranf type + subsample
        self.x = self.x.astype(float)

        #generate train test splits
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, train_size=train_size,
            stratify=self.y)  # split data

        # self.df.to_csv(self.destination)
        # print(f'filtered data saved @: {self.destination}')

        #Fit transform on categorical cols
        encoder = ColumnTransformer(
            [
                ('one_hot_encoder', OneHotEncoder(categories='auto'),
                 self.categorical_features_index)
            ],  
            remainder='passthrough'  # Leave the rest of the columns untouched
        )
        encoder.fit(self.x)  # OHE Categorical Columns
        self.encoded_train = encoder.transform(
            self.X_train)  # Encode Training Set
        self.encoded_test = encoder.transform(self.X_test)
        return self.encoded_train, self.encoded_test, self.y_train, self.y_test, encoder, self.unrolled_features