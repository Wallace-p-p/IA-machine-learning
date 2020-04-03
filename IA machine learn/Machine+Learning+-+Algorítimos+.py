
# coding: utf-8

# In[1]:


#algoritmos machine learning
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# In[2]:


#importando dados de treino
dataTrain = pd.read_csv("exoTrain.csv", sep = ",") #csv treino
label1=  dataTrain.pop(dataTrain.columns[0])
labelTrain= label1.values.flatten()
data1 = dataTrain.values


# In[5]:


#importando dados de teste
dataTest = pd.read_csv("exoTest.csv", sep = ",") #csv teste
label2=  dataTest.pop(dataTest.columns[0])
labelTest= label2.values.flatten()
data2 = dataTest.values


# In[16]:


def MLClassification(data1,data2,labelTrain):

    def Tree(data1,data2,labelTrain):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(data1 , labelTrain)
        PredTree=clf.predict(data2)
        return PredTree
    
    def Svc(data1,data2,labelTrain):
        clf = SVC()
        clf.fit(data1 , labelTrain)
        PredSVC=clf.predict(data2)
        return PredSVC
    
    def KNC(data1,data2,labelTrain):
        neigh = KNeighborsClassifier(n_neighbors=2)
        neigh.fit(data1 , labelTrain)
        PredKNC= neigh.predict(data2)
        return PredKNC
    
    def Bayes(data1,data2,labelTrain):
        clf = GaussianNB()
        clf.fit(data1, labelTrain)
        PredBayes=clf.predict(data2)
        return PredBayes
    
    def Neural(data1,data2,labelTrain):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(data1, labelTrain)
        PredNeural=clf.predict(data2)
        return PredNeural
    
    return(Tree(data1,data2,labelTrain),Svc(data1,data2,labelTrain),KNC(data1,data2,labelTrain),Bayes(data1,data2,labelTrain),Neural(data1,data2,labelTrain))


# In[17]:


comp=MLClassification(data1,data2,labelTrain)


# In[18]:


#previsões e avaliações
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


print(confusion_matrix(labelTest,comp[0]))
print(classification_report(labelTest,comp[0]))


# In[20]:


print(confusion_matrix(labelTest,comp[1]))
print(classification_report(labelTest,comp[1]))


# In[21]:


print(confusion_matrix(labelTest,comp[2]))
print(classification_report(labelTest,comp[2]))


# In[22]:


print(confusion_matrix(labelTest,comp[3]))
print(classification_report(labelTest,comp[3]))


# In[23]:


print(confusion_matrix(labelTest,comp[4]))
print(classification_report(labelTest,comp[4]))

