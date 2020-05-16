import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import sklearn
from IPython.display import HTML, display, display_html


def save_obj(obj, name):
    with open('obj/lendingclub/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/lendingclub/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




class Perturb:
    def __init__(self, gs_object, models):
        self.objects = gs_object
        self.models = models
        self.a = [str(i) + '%' for i in range(50, 151) if i % 10 == 0]
        self.b = [(i / 100) for i in range(50, 151) if i % 10 == 0]
        self.pertu = dict(zip(self.a, self.b))
        self.pert = [i for i in self.pertu.values()]
        self.cat_step = [(i / 100) for i in range(0, 101)
                         if i % 20 == 0]  #Step by sample size: 0:100, 10
            

    def _transform(self, sample_size):
        '''
        Setter for df transformation
        Parms
            sample_size: int, number of observations in sets \in [5000,10000,20000,40000,80000]

        Returns 
            Stores Xtest and ytest holdout sets for perturbation algorithms
        '''
        X_train, X_test, y_train, y_test, encoder, unrolled_features = self.objects.objects[
            sample_size]
        X_test_holdout = X_test.copy()
        np.random.seed(0)
        idx = np.random.choice(X_test_holdout.shape[0], 2000,
                               replace=True)  # Random 2000 samples
        X_test_holdout = X_test_holdout[idx]  # extract

        self.X_test_holdout = pd.DataFrame(
            X_test_holdout.toarray(),
            columns=unrolled_features)  # Convert to DF for column names\

        self.y_test_holdout = y_test[idx]

    def _perturb_continuous_all_models(self, sample_size, mode, feature):
        '''
        perturbation (continuous) helper
        Parms
            sample_size: int, number of observations in sets \in [5000,10000,20000,40000,80000]
            mode: str, 'accuracy' or 'proportion'. proportion renders p(y ==1)
            feature: str, single feature fed as input. Must be selected from continuous feat. subset

        Returns 
            dictionary containing all accuracies/proportion of ones for each estimator.
        '''
        self._transform(sample_size)

        model_names = [self.models[sample_size][i][0] for i in range(3)]
        models = [self.models[sample_size][i][1] for i in range(3)]

        model_preds = {}
        for ind, model in enumerate(models):
            accuracies, num_pos = [], []
            for i in self.pert:
                clone = self.X_test_holdout.copy()
                clone[feature] = self.X_test_holdout[feature] * (i)

                accuracies.append(
                    accuracy_score(self.y_test_holdout,
                                                   model.predict(clone)) * 100)

                num_pos.append(((Counter(model.predict(clone))[1]) /
                                (self.y_test_holdout.shape[0])) * 100)

            model_preds[model_names[ind]] = (accuracies, num_pos)
        return model_preds

    def perturb_graph_continuous_all_models_all(self, sample_size, mode,
                                                feature):
        '''
        Visualization and Table for continuous feature perturbation
        Parms
            sample_size: int, number of observations in sets \in [5000,10000,20000,40000,80000]
            mode: str, 'accuracy' or 'proportion'. proportion renders p(y ==1)
            feature: str, single feature fed as input. Must be selected from continuous feat. subset

        Returns 
            Graph showing 3 estimators with various perturbation ranges.
            Table showing 3 estimators with various perturbation ranges.
        '''
        counter = 0
        perturb_cont = {}
        for sample in sample_size:

            perturb_cont[sample] = self._perturb_continuous_all_models(
                sample, mode=mode, feature=feature)

            fig, ax1 = plt.subplots(1, 1, figsize=(15, 4))

            if 'accuracy' in mode:
                accs = {}
                for k, v in perturb_cont[sample].items():
                    accs[k] = v[0]

                for i in accs.keys():
                    sns.lineplot(x=self.pert, y=accs[i], ax=ax1)
                    ax1.set_ylabel('Accuracy %', fontsize=15)
                    ax1.set_title('Accuracy : {}, Train Size: {}'.format(
                        feature.upper(), sample),
                                  fontsize=25)
                    ax1.set_ylim(0, 100)

            elif 'proportion' in mode:
                props = {}
                for k, v in perturb_cont[sample].items():
                    props[k] = v[1]

                for i in props.keys():
                    sns.lineplot(x=self.pert, y=props[i], ax=ax1)
                    ax1.set_ylabel('% of Predictions == 1', fontsize=15)
                    ax1.set_title(
                        'Proportionality of Predictions :{}, Train Size: {}'.
                        format(feature.upper(), sample),
                        fontsize=25)
                    ax1.set_ylim(0, 100)

            ax1.set_xlabel('{} Perturbation'.format(feature), fontsize=15)
            plt.legend(title='Model',
                       loc='lower right',
                       labels=[
                           'Random Forest', 'Gradient Boosted Classifier',
                           'Logistic Regression'
                       ])
            if 'accuracy' in mode:
                temp = pd.DataFrame(accs).T
            if 'proportion' in mode:
                temp = pd.DataFrame(props).T
            temp.columns = self.a
            if counter == 0:
                print(
                    'Table Showing {} by {} perturbance percentage - 100 % is equiv to the baseline {}%'
                    .format(mode, feature, mode))

            display(HTML(temp.to_html()))
            print('-' * 50)
            counter += 1
        return perturb_cont

    def categorical_perturb_loangrade_overloaded(self, sample_size, column,
                                                 grouping, sub_column,
                                                 subgrouping):
        self._transform(sample_size)


        model_names = [self.models[sample_size][i][0] for i in range(3)]
        models = [self.models[sample_size][i][1] for i in range(3)]

        col = column  #Get Target Column
        cols = [i for i in self.X_test_holdout.columns
                if grouping in i]  #Extract grouping cols

        sub_col = sub_column  #Get Target SubColumn
        sub_cols = [
            i for i in self.X_test_holdout.columns if subgrouping in i
        ]  #Extract Sub-grouping cols

        scores_dict1 = {}
        for m, n in zip(models, model_names):
            nontarget_cols = [x for x in cols
                              if x != col]  #Get all nontarget columns
            nontarget_subcols = [x for x in sub_cols if x != sub_column]

            scores_dict = {}

            for i in self.cat_step:
                clone = self.X_test_holdout.copy(
                )  #Refresh copy each subloop for each model
                scores = []  #Empty
                counter = 0  #Reset Counter for each subloop
                while counter < 10:
                    #Working on overarching Group
                    idx_0s = clone.index[clone[col] ==
                                         0].tolist()  #find all 0 indices
                    idx_1s = clone.index[clone[col] ==
                                         1].tolist()  #find all 1 indices
                    idx = np.random.choice(
                        idx_0s, int(len(idx_1s) * i), replace=True
                    )  #random select n indices from 0 indices (Sample of zero indices at i*size of 1's)

                    clone[col].iloc[idx, ] = 1  #change n indices from 0 to 1

                    for l in nontarget_cols:
                        clone[l].iloc[
                            idx,
                        ] = 0  #change n indices from 1 to 0 if not target column

                    clone[sub_col].iloc[idx,
                                        ] = 1  #change n indices from 0 to 1

                    for o in nontarget_subcols:
                        clone[o].iloc[
                            idx,
                        ] = 0  #change n indices from 1 to 0 if not target column

                    try:
                        scores.append(Counter(m.predict(clone))[1])
                    except:
                        scores.append(Counter(m.predict_classes(clone))[1])
                    counter += 1
                    clone = self.X_test_holdout.copy()
                scores_dict[i] = scores

            scores_dict1[n] = scores_dict

        test_dict_df = pd.DataFrame()
        for i in scores_dict1.keys():
            test_dict_df[i] = pd.DataFrame(scores_dict1[i]).mean().values

        test_dict_df = test_dict_df.T
        #test_dict_df = test_dict_df / 2000
        test_dict_df.columns = self.cat_step
        return test_dict_df

def cat_perturb_plot_helper(model, column, sub_cols, perturbation_object):
    print(f'Perturbing {column} + SubloanGrades {sub_cols}')
    cat_df = pd.DataFrame()
    size = [5000,10000,20000,40000,80000]
    for i in size:
        cat_df_temp = perturbation_object.categorical_perturb_loangrade_overloaded(
            sample_size=i,
            column=column,
            grouping='Loan_Grade',
            sub_column=sub_cols,
            subgrouping='Loan_SubGrade')

        cat_df_temp['sample_size'] = i
        cat_df = cat_df.append(cat_df_temp, sort=False)
        temp = cat_df[cat_df.index == model]
        test = temp[temp['sample_size'] == i].melt('sample_size')
        sns.lineplot(x=perturbation_object.cat_step, y=test.value)
    plt.title(f'Model: {model}, by size of set trained on')
    plt.xlabel = 'perturbation parameter'
    plt.ylabel = 'Number of 1 predictions (2000 samples in holdout)'
    plt.legend(title='Model', loc='lower right', labels=[i for i in size])
    plt.show()
    return temp