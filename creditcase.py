# Usar modelo(s) escolhido(s) sob a base de crédito

#matplotlib inline
#config InlineBackend.figure_format = 'svg'


import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import itertools

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def train_model(tech, n_meses, x):
    # abre os 2 bancos
    data = pd.read_csv("banco/application_record.csv", encoding = 'utf-8') 
    record = pd.read_csv("banco/credit_record.csv", encoding = 'utf-8')  
    
    plt.rcParams['figure.facecolor'] = 'white'
    
    # find all users' account open month.
    begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
    begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'}) 
    new_data=pd.merge(data,begin_month,how="left",on="ID") # merge to record data
    
    # cria coluna 'credit_abuse': se ficou devendo 'n_meses' ou mais em algum
    # momento insere 'Yes'
    record['credit_abuse'] = None
    
    if n_meses >= 1:
        record['credit_abuse'][record['STATUS'] =='1']='Yes' 
    if n_meses >= 2:
        record['credit_abuse'][record['STATUS'] =='2']='Yes'
    if n_meses >= 3:
        record['credit_abuse'][record['STATUS'] =='3']='Yes' 
    if n_meses >= 4:
        record['credit_abuse'][record['STATUS'] =='4']='Yes' 
    if n_meses == 5:
        record['credit_abuse'][record['STATUS'] =='5']='Yes'
    
    # no banco cpunt, classifica como 'Yes' os clientes que atrasaram por
    # 'n_meses' 'x' ou mais vezes
    cpunt=record.groupby('ID').count()
    cpunt['dep_value'] = None
    cpunt['dep_value'][cpunt['credit_abuse'] >= x]='Yes' 
    cpunt['dep_value'][cpunt['credit_abuse'] < x]='No' 
    cpunt = cpunt[['dep_value']]
    new_data=pd.merge(new_data,cpunt,how='inner',on='ID')
    
    # transforma 'target' em 'dep_value'
    new_data['target']=new_data['dep_value']
    
    # transforma 'Yes' em 1 e 'No' em 0
    new_data.loc[new_data['target']=='Yes','target']=1
    new_data.loc[new_data['target']=='No','target']=0
    
    print(cpunt['dep_value'].value_counts())
    cpunt['dep_value'].value_counts(normalize=True)
    
    # renomeia as colunas originais
    new_data.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Realty',\
                             'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',\
                             'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',\
                            'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',\
                             'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',\
                             'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',\
                            'OCCUPATION_TYPE':'occyp'},inplace=True)
    
    # remove linhas com valor NULL
    new_data.dropna()
    new_data = new_data.mask(new_data == 'NULL').dropna()
    
    # remove colunas em namelist da tabela ivtable
    ivtable=pd.DataFrame(new_data.columns,columns=['variable'])
    ivtable['IV']=None
    namelist = ['FLAG_MOBIL','begin_month','dep_value','target','ID']
    
    for i in namelist:
        ivtable.drop(ivtable[ivtable['variable'] == i].index, inplace=True)
        
    # Calculate information value
    def calc_iv(df, feature, target, pr=False):
        lst = []
        df[feature] = df[feature].fillna("NULL")
    
        for i in range(df[feature].nunique()):
            val = list(df[feature].unique())[i]
            lst.append([feature,                                                        # Variable
                        val,                                                            # Value
                        df[df[feature] == val].count()[feature],                        # All
                        df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                        df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)
    
        data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
        data['Share'] = data['All'] / data['All'].sum()
        data['Bad Rate'] = data['Bad'] / data['All']
        data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
        data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
        data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
        
        data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    
        data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    
        data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
        data.index = range(len(data.index))
    
        if pr:
            print(data)
            print('IV = ', data['IV'].sum())
    
        iv = data['IV'].sum()
        print('This variable\'s IV is:',iv)
        print(df[feature].value_counts())
        return iv, data
    
    def convert_dummy(df, feature,rank=0):
        pos = pd.get_dummies(df[feature], prefix=feature)
        mode = df[feature].value_counts().index[rank]
        biggest = feature + '_' + str(mode)
        pos.drop([biggest],axis=1,inplace=True)
        df.drop([feature],axis=1,inplace=True)
        df=df.join(pos)
        return df
    
    def get_category(df, col, binsnum, labels, qcut = False):
        if qcut:
            localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut
        else:
            localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut
            
        localdf = pd.DataFrame(localdf)
        name = 'gp' + '_' + col
        localdf[name] = localdf[col]
        df = df.join(localdf[name])
        df[name] = df[name].astype(object)
        return df
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    # Formatação da tabela para uso no treinamento do modelo
    # print com contagem de valores
    # Cálculo de valor da informação
    
    # Gênero (mulher - 0, homem - 1)
    new_data['Gender'] = new_data['Gender'].replace(['F','M'],[0,1])
    print(new_data['Gender'].value_counts())
    iv, data = calc_iv(new_data,'Gender','target')
    ivtable.loc[ivtable['variable']=='Gender','IV']=iv
    data.head()
    
    # Possui carro (não possui - 0, possui - 1)
    new_data['Car'] = new_data['Car'].replace(['N','Y'],[0,1])
    print(new_data['Car'].value_counts())
    iv, data=calc_iv(new_data,'Car','target')
    ivtable.loc[ivtable['variable']=='Car','IV']=iv
    data.head()
    
    # Possui imóvel (não possui - 0, possui - 1)
    new_data['Realty'] = new_data['Realty'].replace(['N','Y'],[0,1])
    print(new_data['Realty'].value_counts())
    iv, data=calc_iv(new_data,'Realty','target')
    ivtable.loc[ivtable['variable']=='Realty','IV']=iv
    data.head()
    
    # Possui telefone (não possui - 0, possui - 1)
    new_data['phone']=new_data['phone'].astype(str)
    print(new_data['phone'].value_counts(normalize=True,sort=False))
    new_data.drop(new_data[new_data['phone'] == 'nan' ].index, inplace=True)
    iv, data=calc_iv(new_data,'phone','target')
    ivtable.loc[ivtable['variable']=='phone','IV']=iv
    data.head()
    
    # Possui e-mail (não possui - 0, possui - 1)
    print(new_data['email'].value_counts(normalize=True,sort=False))
    new_data['email']=new_data['email'].astype(str)
    iv, data=calc_iv(new_data,'email','target')
    ivtable.loc[ivtable['variable']=='email','IV']=iv
    data.head()
    
    # Possui telefone profissional (não possui - 0, possui - 1)
    new_data['wkphone']=new_data['wkphone'].astype(str)
    iv, data = calc_iv(new_data,'wkphone','target')
    new_data.drop(new_data[new_data['wkphone'] == 'nan' ].index, inplace=True)
    ivtable.loc[ivtable['variable']=='wkphone','IV']=iv
    data.head()
    
    # Número de filhos (nas faixas: 0; 1; 2 ou mais)
    new_data.loc[new_data['ChldNo'] >= 2,'ChldNo']='2More'
    print(new_data['ChldNo'].value_counts(sort=False))
    iv, data=calc_iv(new_data,'ChldNo','target')
    ivtable.loc[ivtable['variable']=='ChldNo','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'ChldNo')
    
    # Renda anual (transformada em 3 categorias: low, medium, high)
    new_data['inc']=new_data['inc'].astype(np.int64)
    new_data['inc'] = new_data['inc']/10000
    print(new_data['inc'].value_counts(bins=10,sort=False))
    new_data['inc'].plot(kind='hist',bins=50,density=True)
    new_data = get_category(new_data,'inc', 3, ["low","medium", "high"], qcut = True)
    iv, data = calc_iv(new_data,'gp_inc','target')
    ivtable.loc[ivtable['variable']=='inc','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'gp_inc')
    
    # Idade (transformada em 5 categorias: lowest, low, medium, high, highest)
    new_data['Age']=-(new_data['DAYS_BIRTH'])//365	
    print(new_data['Age'].value_counts(bins=10,normalize=True,sort=False))
    new_data['Age'].plot(kind='hist',bins=20,density=True)
    new_data = get_category(new_data,'Age',5, ["lowest","low","medium","high","highest"])
    iv, data = calc_iv(new_data,'gp_Age','target')
    ivtable.loc[ivtable['variable']=='DAYS_BIRTH','IV'] = iv
    data.head()
    new_data = convert_dummy(new_data,'gp_Age')
    
    # Tempo no emprego atual (transformado em 5 categorias: lowest, low, medium, high, highest)
    new_data['worktm']=-(new_data['DAYS_EMPLOYED'])//365	
    new_data[new_data['worktm']<0] = np.nan # replace by na
    new_data['DAYS_EMPLOYED']
    new_data['worktm'].fillna(new_data['worktm'].mean(),inplace=True) #replace na by mean
    new_data['worktm'].plot(kind='hist',bins=20,density=True)
    new_data = get_category(new_data,'worktm',5, ["lowest","low","medium","high","highest"])
    iv, data=calc_iv(new_data,'gp_worktm','target')
    ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'gp_worktm')
    
    # Tamanho da família (em 3 faixas: 1; 2; 3 ou mais)
    new_data['famsize'].value_counts(sort=False)
    new_data['famsize']=new_data['famsize'].astype(int)
    new_data['famsizegp']=new_data['famsize']
    new_data['famsizegp']=new_data['famsizegp'].astype(object)
    new_data.loc[new_data['famsizegp']>=3,'famsizegp']='3more'
    iv, data=calc_iv(new_data,'famsizegp','target')
    ivtable.loc[ivtable['variable']=='famsize','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'famsizegp')
    
    # Tipo de renda (pensionista e estudante agrupados com servidor público)
    print(new_data['inctp'].value_counts(sort=False))
    print(new_data['inctp'].value_counts(normalize=True,sort=False))
    new_data.loc[new_data['inctp']=='Pensioner','inctp']='State servant'
    new_data.loc[new_data['inctp']=='Student','inctp']='State servant'
    iv, data=calc_iv(new_data,'inctp','target')
    ivtable.loc[ivtable['variable']=='inctp','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'inctp')
    
    # Tipo de ocupação (transformados em 3 grupos: laborista, de escritório, e high tech)
    new_data.loc[(new_data['occyp']=='Cleaning staff') | (new_data['occyp']=='Cooking staff') | (new_data['occyp']=='Drivers') | (new_data['occyp']=='Laborers') | (new_data['occyp']=='Low-skill Laborers') | (new_data['occyp']=='Security staff') | (new_data['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
    new_data.loc[(new_data['occyp']=='Accountants') | (new_data['occyp']=='Core staff') | (new_data['occyp']=='HR staff') | (new_data['occyp']=='Medicine staff') | (new_data['occyp']=='Private service staff') | (new_data['occyp']=='Realty agents') | (new_data['occyp']=='Sales staff') | (new_data['occyp']=='Secretaries'),'occyp']='officewk'
    new_data.loc[(new_data['occyp']=='Managers') | (new_data['occyp']=='High skill tech staff') | (new_data['occyp']=='IT staff'),'occyp']='hightecwk'
    print(new_data['occyp'].value_counts())
    iv, data=calc_iv(new_data,'occyp','target')
    ivtable.loc[ivtable['variable']=='occyp','IV']=iv
    data.head()         
    new_data = convert_dummy(new_data,'occyp')
    
    # Tipo de moradia
    iv, data=calc_iv(new_data,'houtp','target')
    ivtable.loc[ivtable['variable']=='houtp','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'houtp')
    
    # Nível de escolaridade (mestrado/doutorado agrupado com ensino superior)
    new_data.loc[new_data['edutp']=='Academic degree','edutp']='Higher education'
    iv, data=calc_iv(new_data,'edutp','target')
    ivtable.loc[ivtable['variable']=='edutp','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'edutp')
    
    # Estado civil
    new_data['famtp'].value_counts(normalize=True,sort=False)
    iv, data=calc_iv(new_data,'famtp','target')
    ivtable.loc[ivtable['variable']=='famtp','IV']=iv
    data.head()
    new_data = convert_dummy(new_data,'famtp')
    
    
    ivtable=ivtable.sort_values(by='IV',ascending=False)
    ivtable.loc[ivtable['variable']=='DAYS_BIRTH','variable']='agegp'
    ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','variable']='worktmgp'
    ivtable.loc[ivtable['variable']=='inc','variable']='incgp'
    ivtable
    
    new_data.columns
    
    Y = new_data['target']
    X = new_data[['Gender','Realty','ChldNo_1', 'ChldNo_2More','wkphone',
                  'gp_Age_high', 'gp_Age_highest', 'gp_Age_low',
           'gp_Age_lowest','gp_worktm_high', 'gp_worktm_highest',
           'gp_worktm_low', 'gp_worktm_medium','occyp_hightecwk', 
                  'occyp_officewk','famsizegp_1', 'famsizegp_3more',
           'houtp_Co-op apartment', 'houtp_Municipal apartment',
           'houtp_Office apartment', 'houtp_Rented apartment',
           'houtp_With parents','edutp_Higher education',
           'edutp_Incomplete higher', 'edutp_Lower secondary','famtp_Civil marriage',
           'famtp_Separated','famtp_Single / not married','famtp_Widow']]
    
    Y = Y.astype('int')
    X_balance,Y_balance = SMOTE().fit_resample(X,Y)
    X_balance = pd.DataFrame(X_balance, columns = X.columns)
    
    # Separação do banco em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_balance,Y_balance, 
                                                        stratify=Y_balance, test_size=0.3,
                                                        random_state = 10086)
    
    model = ""
    y_predict = ""
    sns.set_style('white') 
    class_names = ['0','1']
    
    # Seleção do modelo a ser usado
    
    # Regressão logística
    if tech == 0:
        model = LogisticRegression(C=0.8,
                                   random_state=0,
                                   solver='lbfgs')
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))    
        plot_confusion_matrix(confusion_matrix(y_test,y_predict),
                              classes= class_names, normalize = True, 
                              title='Normalized Confusion Matrix: Logistic Regression')
    
    # Árvore de decisão
    elif tech == 1:
        model = DecisionTreeClassifier(max_depth=12,
                                       min_samples_split=8,
                                       random_state=1024)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))    
        plot_confusion_matrix(confusion_matrix(y_test,y_predict),
                              classes=class_names, normalize = True, 
                              title='Normalized Confusion Matrix: CART')
    
    # Random Forest
    elif tech == 2:
        model = RandomForestClassifier(n_estimators=250,
                                      max_depth=12,
                                      min_samples_leaf=16)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
        plot_confusion_matrix(confusion_matrix(y_test,y_predict),
                              classes=class_names, normalize = True, 
                              title='Normalized Confusion Matrix: Ramdom Forests')
        
    # SVM
    elif tech == 3:
        model = svm.SVC(C = 0.8,
                        kernel='linear')
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
        plot_confusion_matrix(confusion_matrix(y_test,y_predict),
                              classes=class_names, normalize = True, 
                              title='Normalized Confusion Matrix: SVM')
        
    # LGBM
    elif tech == 4:
        model = LGBMClassifier(num_leaves=31,
                               max_depth=8, 
                               learning_rate=0.02,
                               n_estimators=250,
                               subsample = 0.8,
                               colsample_bytree =0.8)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
    
    # XGBoost
    elif tech == 5:
        def plot_importance(classifer, x_train, point_size = 25):
            #plot feature importance
            values = sorted(zip(x_train.columns, classifer.feature_importances_), key = lambda x: x[1] * -1)
            imp = pd.DataFrame(values,columns = ["Name", "Score"])
            imp.sort_values(by = 'Score',inplace = True)
            sns.scatterplot(x = 'Score',y='Name', linewidth = 0,
                        data = imp,s = point_size, color='red').set(
            xlabel='importance', 
            ylabel='features')
        plot_importance(model, X_train,20)       
        model.booster_.feature_importance(importance_type='gain')
        
        model = XGBClassifier(max_depth=12,
                              n_estimators=250,
                              min_child_weight=8, 
                              subsample=0.8, 
                              learning_rate =0.02,    
                              seed=42)    
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
        plot_importance(model, X_train, 20)
    
    # Cat Boost
    elif tech == 6:
        model = CatBoostClassifier(iterations=250,
                                   learning_rate=0.2,
                                   od_type='Iter',
                                   verbose=25,
                                   depth=16,
                                   random_seed=42)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
    
    return model, X_train, y_train, X_test, y_test, y_predict



def predict(model, nf, rn, dn, da, ni, re, es, ec, tm, to, gn, car, rp, ce, tp, tf, em):
    '''X = new_data[['Gender','Realty','ChldNo_1', 'ChldNo_2More','wkphone',
                  'gp_Age_high', 'gp_Age_highest', 'gp_Age_low',
           'gp_Age_lowest','gp_worktm_high', 'gp_worktm_highest',
           'gp_worktm_low', 'gp_worktm_medium','occyp_hightecwk', 
                  'occyp_officewk','famsizegp_1', 'famsizegp_3more',
           'houtp_Co-op apartment', 'houtp_Municipal apartment',
           'houtp_Office apartment', 'houtp_Rented apartment',
           'houtp_With parents','edutp_Higher education',
           'edutp_Incomplete higher', 'edutp_Lower secondary','famtp_Civil marriage',
           'famtp_Separated','famtp_Single / not married','famtp_Widow']]'''
                                
    params = []
    params += [gn, re]
    if nf == 1:
        params += [1, 0]
    elif nf > 1:
        params += [0, 1]
    else:
        params += [0, 0]
        
    if tp == "true":
        params += [1]
    else:
        params += [0]
    
    dn = datetime.datetime(int(dn[6:]), int(dn[3:5]), int(dn[0:2]))
    dlowest = datetime.datetime(2005, 1, 1)
    dlow = datetime.datetime(2000, 1, 1)
    dhigh = datetime.datetime(1950, 1, 1)
    dhighest = datetime.datetime(1930, 1, 1)
    if dn < dhighest:
        params += [0, 1, 0, 0]
    elif dn < dhigh:
        params += [1, 0, 0, 0]
    elif dn < dlow:
        params += [0, 0, 1, 0]
    elif dn < dlowest:
        params += [0, 0, 0, 1]
    else:
        params += [0, 0, 0, 0]
    
    da = datetime.datetime(int(da[6:]), int(da[3:5]), int(da[0:2]))
    dlow = datetime.datetime(2005, 1, 1)
    dmedium = datetime.datetime(2000, 1, 1)
    dhigh = datetime.datetime(1981, 1, 1)
    dhighest = datetime.datetime(1930, 1, 1)
    if da < dhighest:
        params += [0, 1, 0, 0]
    elif da < dhigh:
        print('veio')
        params += [1, 0, 0, 0]
    elif da < dlow:
        params += [0, 0, 1, 0]
    elif da < dmedium:
        params += [0, 0, 0, 1]
    else:
        params += [0, 0, 0, 0]
        
    if to >= 15 and to <= 17:
        params += [1, 0]
    elif to >= 7 and to <= 14:
        params += [0, 1]
    else:
        params += [0, 0]
    
    if ni == 1:
        params += [1, 0]
    elif ni == 2:
        params += [0, 0]
    else:
        params += [0, 1]
    
    if tm == 0:
        params += [1, 0, 0, 0, 0]
    elif tm == 1:
        params += [0, 0, 0, 0, 1]
    elif tm == 2:
        params += [0, 1, 0, 0, 0]
    elif tm == 3:
        params += [0, 0, 0, 1, 0]
    elif tm == 4:
        params += [0, 0, 1, 0, 0]
    else:
        params += [0, 0, 0, 0, 0]
    
    if es == 1:
        params += [1, 0, 0]
    elif es == 2:
        params += [0, 1, 0]
    elif es == 3:
        params += [0, 0, 1]
    else:
        params += [0, 0, 0]
        
    if ec == 1:
        params += [0, 0, 1, 0]
    elif ec == 2:
        params += [1, 0, 0, 0]
    elif ec == 3:
        params += [0, 1, 0, 0]
    elif ec == 4:
        params += [0, 0, 0, 1]
    else:
        params += [0, 0, 0, 0]
  
    return model.predict([params])

#retrain_model("", 0, 0)