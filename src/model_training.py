from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import preprocessing

from joblib import dump, load

def model_training(data_path):
    x_scaled,y_encode=preprocessing(data_path)
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_encode,test_size=0.2,random_state=42,stratify=y_encode)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return dump(model, 'random_forest_model.joblib')
   
    


    
