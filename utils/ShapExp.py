
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from utils.Dataloader import check_existence


class ExplainShap():
    
    def __init__(self, gs_object, models):
        self.gs_object = gs_object
        self.models = models
        self.shap = generate_shap_values(gs_object=self.gs_object, model_dict=self.models)
        self.class_names = ['Charged Off', 'Fully Paid']
        
        

    def shap_many_graph(self, model, num_obs):
        '''
        Returns:
        Global Shap Explanations over test set
        '''
        X_train, X_test, y_train, y_test, encoder, unrolled_features = self.gs_object.objects[num_obs]
        
        X_train = pd.DataFrame(X_train.toarray(), columns= unrolled_features)
        X_test = pd.DataFrame(X_test.toarray(), columns = unrolled_features)
        
        estimator = [x[1] for x in self.models[num_obs] if x[0] == model][0]
        
        f = lambda x: estimator.predict_proba(x)[:, 1]
        
        med = X_train.median().values.reshape((1, X_train.shape[1]))
        
        explainer = shap.KernelExplainer(f, med)
        return shap.force_plot(explainer.expected_value, self.shap[num_obs][model],
                               X_test)

    def shap_summary_graph(self, model, num_obs):
        '''
        Returns:
        Global Shap Explanations over test set - Summary
        '''
        X_train, X_test, y_train, y_test, encoder, unrolled_features = self.gs_object.objects[num_obs]
        
        X_train = pd.DataFrame(X_train.toarray(), columns= unrolled_features)
        X_test = pd.DataFrame(X_test.toarray(), columns = unrolled_features)
        
        estimator = [x[1] for x in self.models[num_obs] if x[0] == model][0]
        
        f = lambda x: estimator.predict_proba(x)[:, 1]
        
        med = X_train.median().values.reshape((1, X_train.shape[1]))
        
        explainer = shap.KernelExplainer(f, med)
        
        print(f"{model} Shap Values")
        #Try-catch to deal with some irrelevant error in shap source code
        try:
            return shap.summary_plot(self.shap[num_obs][model],
                                     features = X_test,
                                     feature_names= unrolled_features,
                                     class_names=self.class_names)
        except:
            pass



def save_abs_shap_values_sample(sample_size, plot_obj, gs, models = ['RFC', 'GBC', 'LR']):
    
    shap_df = pd.DataFrame()
    X_train, X_test, y_train, y_test, encoder, unrolled_features = gs.objects[sample_size]
    
    for model_n in gs.model_names:
        df = pd.DataFrame.from_dict(plot_obj.shap[sample_size][model_n])
        df.columns = unrolled_features
        df = abs(df) 
        df['model'] = model_n
        df['sample_size'] = sample_size
        df.set_index(['model', 'sample_size'], inplace=True)
        shap_df = shap_df.append(df)
    file_name = f'./shap_values/sv_abs_{sample_size}.csv'
    if not os.path.isfile(file_name):
        shap_df.to_csv(file_name)
        print(f'file saved @ {file_name}')
    else:
        print(f'files previously generated. Reference {file_name} ')
    return shap_df


#----------------------------------------------------------------
#Shap Gen. Helpers
#----------------------------------------------------------------


def create_dir(verbose=True):
    dirName = os.getcwd() + '\\shap_values'
    try:
        os.mkdir(dirName)
        if verbose:
            print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists ")


def save_shap_values(svs, verbose=True):
    create_dir(verbose=True)
    file_name = './shap_values/all_shap_values.pkl'
    pickle.dump(svs, open(file_name, 'wb'))
    if verbose:
        print(f'shap values saved @ {file_name}')


def generate_shap_values(gs_object, model_dict):
    '''
    Generate Shap Attribution values for all models
        Parms
            gs_object; gridsearch object instantiated during model construction.
                       This stores the train test splits for various sized datasets

            models; pass in model_dict from model construction
        Returns
            A dictionary with all shap values for all models of all sample sizes
    '''
    file_name = './shap_values/all_shap_values.pkl'
    if not check_existence(file_name):
        shap_values_outer = {}
        size = [5000,10000,20000,40000,80000]

        for i in size:
            shap_values_inner = {}
            X_train, X_test, y_train, y_test, encoder, unrolled_features = gs_object.objects[i] #Get sets for ith size dataset
            X_train_shap = pd.DataFrame(X_train.toarray(), columns= unrolled_features)
            X_test_shap = pd.DataFrame(X_test.toarray(), columns = unrolled_features)

            model_names =  [j[0] for j in model_dict[i]]
            models = [j[1] for j in model_dict[i]]

            for m,n in zip(model_names, models):
                f = lambda x: n.predict_proba(x)[:, 1]
                med = X_train_shap.median().values.reshape((1, X_train_shap.shape[1]))
                explainer = shap.KernelExplainer(f, med)
                shap_values = explainer.shap_values(X_test_shap, samples =1000, l1_reg = 'aic')
                shap_values_inner[m] = shap_values
            shap_values_outer[i] = shap_values_inner
        save_shap_values(svs = shap_values_outer, verbose=True)
    else:
        print('shap values already generated')
        file = open(file_name,'rb')
        shap_values_outer = pickle.load(file)

    return shap_values_outer

def plot_abs_shap_values(directory='./shap_values/',
                         sample_size=5000,
                         num_feats=10,
                         normalized=True):
    df = pd.read_csv(directory + f'sv_abs_{sample_size}.csv')
    for model in ['RFC', 'GBC', 'LR']:
        temp = df[df['model'] == model]
        temp.set_index(['model', 'sample_size'], inplace=True)
        temp = pd.DataFrame(temp.sum(axis=0)).reset_index()
        temp.columns = ['feature', 'svs']
        temp.set_index('feature', inplace=True)
        if normalized:
            temp = temp.div(temp.sum(axis=0), axis=1)
        temp = temp.nlargest(num_feats, 'svs')
        sns.barplot(x=temp.svs, y=temp.index)
        plt.title(
            f'Sum of Absolute Shap Values. Model: {model}, sample_size: {sample_size}, normalized: {normalized}'
        )
        plt.show()