import pandas as pd
import numpy as np
import time

import pickle

import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from random_forest import RandomForest


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, features_path, labels_path):
        print(f"Loading data from {features_path} and {labels_path}...")
        
        x = pd.read_csv(features_path).values
        y = pd.read_csv(labels_path).squeeze().values
        
        test_size = self.config['test_size']
        random_state = self.config['random_state']
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training samples: {len(self.x_train)}, Test samples: {len(self.x_test)}")
        
    def train(self):
        print("Training model...")
        start_time = datetime.now()
        
        self.model = RandomForest(self.x_train, self.y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        
    def evaluate(self):
        print("Evaluating model...")
        
        train_preds = self.model.predict(self.x_train)
        test_preds = self.model.predict(self.x_test)
        
        train_acc = np.mean(train_preds == self.y_train)
        test_acc = np.mean(test_preds == self.y_test)
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:     {test_acc:.4f}")
        
    def save_model(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {output_path}!")
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {model_path}!")
        
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_from_csv(self, csv_path, output_path=None):
        print(f"Loading data from {csv_path}...")
        x = pd.read_csv(csv_path).values
        
        predictions = self.predict(x)
        
        if output_path:
            pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return predictions


# Create ArgumentParser for CLI
parser = argparse.ArgumentParser(description='Tennis Prediction Engine - CLI')

parser.add_argument('--mode', type=str, required=True,
                    choices=['train', 'evaluate', 'predict'],
                    help='Operation mode')
parser.add_argument('--features', type=str, help='Path to features CSV file')
parser.add_argument('--labels', type=str, help='Path to labels CSV')
parser.add_argument('--model', type=str, help='Path to saved model if evaluating/predicting')
parser.add_argument('--test-size', type=float, default=0.2, help='How much of the data to use as the test set')
parser.add_argument('--random-state', type=int, default=42, help='Pseudorandom seed')

# Read arguments
args = parser.parse_args()

# Generate config dictionary from args to pass into the ModelTrainer
config = {
    'test_size': args.test_size,
    'random_state': args.random_state
}

trainer = ModelTrainer(config)

if args.mode == 'train':
    trainer.load_data(args.features, args.labels)
    trainer.train()
    trainer.evaluate()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(f"models/model_{timestamp}")
elif args.mode == 'evaluate':
    trainer.load_model(args.model)
    trainer.load_data(args.features, args.labels)
    start = time.time()
    trainer.evaluate()
    print(f'Time elapsed: {time.time() - start}')
elif args.mode == 'predict':
    trainer.load_model(args.model)
    trainer.predict_from_csv(args.features, args.output)