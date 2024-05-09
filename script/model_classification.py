from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn .metrics import accuracy_score
from sklearn.svm import SVC
import joblib
import numpy as np

from skimage import feature
from skimage.transform import resize
import cv2

class SVMObjectClassifier():
    
    def __init__(self, kernel='rbf', C=0.5, random_state=42, probability=True):
        self.model = SVC(kernel=kernel, C=C, random_state=random_state, probability=probability)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extracter = None
    
    def get_feature_extracter(self, feature_extracter):
        self.feature_extracter = feature_extracter
    
    def prepare_dataset(self, X, y):
        X_features = []
        for x in X:
            hog_features = self.feature_extracter.features(x)
            X_features.append(hog_features)
        X_features = np.array(X_features)
        X_features = self.scaler.fit_transform(X_features)
        y_encoded = self.label_encoder.fit_transform(y)
        print(X_features.shape, y_encoded.shape)
        return X_features, y_encoded
    
    def train(self, X, y):
        X_features, y_encoded = self.prepare_dataset(X, y)
        self.model.fit(X_features, y_encoded)
    
    def save(self, path):
        joblib.dump(self.model, path + '/model.pkl')
        joblib.dump(self.scaler, path + '/scaler.pkl')
        joblib.dump(self.label_encoder, path + '/label_encoder.pkl')
    
    def load(self, path):
        self.model = joblib.load(path + '/model.pkl')
        self.scaler = joblib.load(path + '/scaler.pkl')
        self.label_encoder = joblib.load(path + '/label_encoder.pkl')
            
    def evaluate(self, X, y):
        X_features = []
        for x in X:
            hog_features = self.feature_extracter.features(x)
            X_features.append(hog_features)
        X_features = np.array(X_features)
        X_features = self.scaler.transform(X_features)
        y_encoded = self.label_encoder.transform(y)
        y_pred = self.model.predict(X_features)
        accuracy = accuracy_score(y_encoded, y_pred)
        print(f'Accuracy on validation dataset: {accuracy:.2f}')
        return accuracy
    
    def predict(self, X):
        X_features = []
        for x in X:
            x = cv2.resize(x, (64, 64))
            hog_features = self.feature_extracter.features(x)
            X_features.append(hog_features)
        X_features = np.array(X_features)
        X_id = np.arange(0, len(X_features))
        X_features = self.scaler.transform(X_features)
        y_pred_prob = self.model.predict_proba(X_features)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred_prob[X_id, y_pred], self.label_encoder.inverse_transform(y_pred)