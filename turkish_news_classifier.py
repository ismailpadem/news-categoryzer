import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from TurkishStemmer import TurkishStemmer
import os
import copy
import random
import time
import json
import datetime
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class TextAugmentation:
    """Text augmentation techniques for Turkish news data"""
    
    def __init__(self, aug_prob=0.3, max_aug_per_text=2):
        self.aug_prob = aug_prob  
        self.max_aug_per_text = max_aug_per_text  
        
        # Common Turkish words for insertion
        self.common_words = [
            've', 'bir', 'bu', 'olan', 'için', 'ile', 'da', 'de', 'var', 'olan',
            'çok', 'daha', 'en', 'ama', 'ancak', 'sonra', 'şimdi', 'bugün',
            'yeni', 'büyük', 'önemli', 'iyi', 'güzel', 'başka', 'tüm', 'bütün'
        ]
        
        
        self.stop_words = [
            've', 'bir', 'bu', 'şu', 'o', 'da', 'de', 'ki', 'mi', 'mu', 'mü',
            'için', 'ile', 'gibi', 'kadar', 'daha', 'en', 'çok', 'az', 'bile'
        ]
    
    def random_deletion(self, text, delete_prob=0.1):
        """Randomly delete words from text"""
        words = text.split()
        if len(words) <= 2:
            return text
        
        new_words = []
        for word in words:
            if random.random() > delete_prob:
                new_words.append(word)
        
        if len(new_words) == 0:
            return words[random.randint(0, len(words)-1)]
        
        return ' '.join(new_words)
    
    def random_swap(self, text, swap_prob=0.1):
        """Randomly swap words in text"""
        words = text.split()
        if len(words) <= 2:
            return text
        
        new_words = words.copy()
        for i in range(len(words)):
            if random.random() < swap_prob:
                swap_idx = random.randint(0, len(words)-1)
                new_words[i], new_words[swap_idx] = new_words[swap_idx], new_words[i]
        
        return ' '.join(new_words)
    
    def random_insertion(self, text, insert_prob=0.1):
        """Randomly insert common words into text"""
        words = text.split()
        new_words = words.copy()
        
        for i in range(len(words)):
            if random.random() < insert_prob:
                random_word = random.choice(self.common_words)
                new_words.insert(i, random_word)
        
        return ' '.join(new_words)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with similar Turkish words (simplified version)"""
        words = text.split()
        new_words = words.copy()
        
        
        synonyms = {
            'büyük': ['kocaman', 'iri', 'dev'],
            'küçük': ['minik', 'ufak', 'minyon'],
            'güzel': ['hoş', 'zarif', 'şirin'],
            'kötü': ['fena', 'berbat', 'çirkin'],
            'iyi': ['güzel', 'hoş', 'mükemmel'],
            'yeni': ['taze', 'fresh', 'modern'],
            'eski': ['köhne', 'antika', 'geçmiş'],
            'hızlı': ['çabuk', 'süratli', 'acele'],
            'yavaş': ['ağır', 'sakin', 'durgun']
        }
        
        for i, word in enumerate(words):
            if word.lower() in synonyms and random.random() < replace_prob:
                new_words[i] = random.choice(synonyms[word.lower()])
        
        return ' '.join(new_words)
    
    def back_translation_simulation(self, text):
        """Simulate back translation by shuffling sentence structure"""
        words = text.split()
        if len(words) <= 3:
            return text
        
        
        new_words = words.copy()
        
        
        for i in range(len(words)-1):
            if len(words[i]) > 5 and random.random() < 0.2:
                # Simple heuristic for potential adjective-noun swapping
                new_words[i], new_words[i+1] = new_words[i+1], new_words[i]
        
        return ' '.join(new_words)
    
    def augment_text(self, text):
        """Apply multiple augmentation techniques to a single text"""
        if random.random() > self.aug_prob:
            return text
        
        augmented_texts = [text]
        
        # List of augmentation functions
        aug_functions = [
            self.random_deletion,
            self.random_swap,
            self.random_insertion,
            self.synonym_replacement,
            self.back_translation_simulation
        ]
        
        # Apply random augmentations
        num_augmentations = random.randint(1, self.max_aug_per_text)
        selected_augs = random.sample(aug_functions, min(num_augmentations, len(aug_functions)))
        
        for aug_func in selected_augs:
            augmented_text = aug_func(text)
            if augmented_text != text and len(augmented_text.split()) > 2:
                augmented_texts.append(augmented_text)
        
        return random.choice(augmented_texts)
    
    def augment_dataset(self, texts, labels, augmentation_factor=1.5):
        """Augment the entire dataset"""
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Keep original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Add augmented versions
            num_augmentations = int(augmentation_factor)
            for _ in range(num_augmentations):
                aug_text = self.augment_text(text)
                if aug_text != text:  # Only add if actually changed
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True, monitor='val_f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        # Determine if higher is better
        self.higher_is_better = monitor in ['val_f1', 'val_acc', 'val_accuracy']
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"🔄 Restored best weights (best {self.monitor}: {self.best_score:.4f})")
    
    def _is_improvement(self, score):
        if self.higher_is_better:
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class TextPreprocessor:
    """Text preprocessing utilities for Turkish news data"""
    
    def __init__(self):
        self.stemmer = TurkishStemmer()
    
    def normalize_text(self, text):
        """Normalize text by removing HTML, punctuation, numbers and extra whitespaces"""
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text):
        """Lemmatize Turkish text using TurkishStemmer"""
        words = text.split()
        lemmatized_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(lemmatized_words)


class Vocabulary:
    """Vocabulary class for text tokenization"""
    
    def __init__(self, max_vocab_size=10000, oov_token='<OOV>'):
        self.max_vocab_size = max_vocab_size
        self.oov_token = oov_token
        self.word_to_idx = {oov_token: 0}
        self.idx_to_word = {0: oov_token}
        self.word_freq = Counter()
    
    def fit_on_texts(self, texts):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Keep only most frequent words
        most_common = self.word_freq.most_common(self.max_vocab_size - 1)
        
        # Build word-to-index mapping
        for idx, (word, freq) in enumerate(most_common, 1):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            words = text.split()
            sequence = [self.word_to_idx.get(word, 0) for word in words]
            sequences.append(sequence)
        return sequences
    
    def save(self, filepath):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'max_vocab_size': self.max_vocab_size,
            'oov_token': self.oov_token
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load(self, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.oov_token = vocab_data['oov_token']


def pad_sequences(sequences, max_length=200, padding='post', truncating='post', value=0):
    """Pad sequences to same length"""
    padded = []
    for seq in sequences:
        if len(seq) > max_length:
            if truncating == 'post':
                seq = seq[:max_length]
            else:
                seq = seq[-max_length:]
        
        if len(seq) < max_length:
            if padding == 'post':
                seq = seq + [value] * (max_length - len(seq))
            else:
                seq = [value] * (max_length - len(seq)) + seq
        
        padded.append(seq)
    
    return np.array(padded)


class NewsDataset(Dataset):
    """PyTorch Dataset for Turkish news data"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for Turkish news categorization"""
    
    def __init__(self, vocab_size, embedding_dim=1024, hidden_dim=1024, 
                 num_classes=7, dropout_rate=0.5, max_length=200):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim * 2, 32, batch_first=True, 
                            bidirectional=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(32 * 2, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)
        
        # First BiLSTM layer
        lstm1_out, _ = self.lstm1(embedded)
        
        # Second BiLSTM layer
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out)
        
        # Use the last hidden state
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # We want the last layer's hidden state from both directions
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(last_hidden))
        dropped = self.dropout(fc1_out)
        output = self.fc2(dropped)
        
        return output


class TurkishNewsClassifier:
    """Main class for Turkish news classification"""
    
    def __init__(self, max_vocab_size=10000, max_length=200, embedding_dim=1024,
                 hidden_dim=1024, dropout_rate=0.5, learning_rate=0.001, 
                 use_augmentation=True, augmentation_factor=1.5):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_augmentation = use_augmentation
        self.augmentation_factor = augmentation_factor
        
        self.preprocessor = TextPreprocessor()
        self.augmenter = TextAugmentation() if use_augmentation else None
        self.vocabulary = Vocabulary(max_vocab_size)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results tracking
        self.results = {
            'training_start_time': None,
            'training_end_time': None,
            'training_duration': None,
            'dataset_info': {},
            'model_config': {},
            'training_history': {},
            'test_results': {},
            'detailed_metrics': {},
            'confusion_matrix': None,
            'classification_report': None
        }
        
        print(f"Using device: {self.device}")
        if use_augmentation:
            print(f"🔄 Data augmentation enabled (factor: {augmentation_factor})")
        else:
            print("📊 Data augmentation disabled")
    
    def load_data(self, csv_path):
        """Load and explore the dataset"""
        self.df = pd.read_csv(csv_path)
        
        # Store dataset info in results
        self.results['dataset_info'] = {
            'csv_path': csv_path,
            'total_samples': len(self.df),
            'shape': self.df.shape,
            'categories': self.df['category'].unique().tolist(),
            'category_distribution': self.df['category'].value_counts().to_dict(),
            'load_time': datetime.datetime.now().isoformat()
        }
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Categories: {self.df['category'].unique()}")
        print(f"Category distribution:\n{self.df['category'].value_counts()}")
        return self.df
    
    def preprocess_data(self):
        """Preprocess the text data"""
        preprocessing_start = time.time()
        print("Preprocessing text data...")
        
        # Normalize text
        self.df['normalized_text'] = self.df['text'].apply(self.preprocessor.normalize_text)
        
        # Lemmatize text
        self.df['lemmatized_text'] = self.df['normalized_text'].apply(
            self.preprocessor.lemmatize_text)
        
        original_size = len(self.df)
        
        # Apply data augmentation if enabled
        if self.use_augmentation:
            print(f"🔄 Applying data augmentation (factor: {self.augmentation_factor})...")
            
            # Extract texts and labels for augmentation
            texts = self.df['lemmatized_text'].tolist()
            categories = self.df['category'].tolist()
            
            # Apply augmentation
            augmented_texts, augmented_categories = self.augmenter.augment_dataset(
                texts, categories, self.augmentation_factor)
            
            # Create new dataframe with augmented data
            augmented_df = pd.DataFrame({
                'category': augmented_categories,
                'text': augmented_texts,
                'normalized_text': augmented_texts,  # Already processed
                'lemmatized_text': augmented_texts   # Already processed
            })
            
            # Replace original dataframe
            self.df = augmented_df
            
            new_size = len(self.df)
            print(f"✅ Dataset augmented: {original_size} → {new_size} samples ({new_size/original_size:.1f}x increase)")
            
            # Show category distribution after augmentation
            print("📊 Augmented category distribution:")
            print(self.df['category'].value_counts().to_string())
        
        # Build vocabulary and convert to sequences
        self.vocabulary.fit_on_texts(self.df['lemmatized_text'])
        sequences = self.vocabulary.texts_to_sequences(self.df['lemmatized_text'])
        
        # Pad sequences
        self.padded_sequences = pad_sequences(sequences, max_length=self.max_length)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(self.df['category'])
        self.num_classes = len(self.label_encoder.classes_)
        
        # Convert to one-hot encoding
        self.categorical_labels = np.eye(self.num_classes)[encoded_labels]
        
        preprocessing_time = time.time() - preprocessing_start
        
        # Update results with preprocessing info
        self.results['dataset_info'].update({
            'original_size': original_size,
            'final_size': len(self.df),
            'augmentation_used': self.use_augmentation,
            'augmentation_factor': self.augmentation_factor if self.use_augmentation else None,
            'augmented_category_distribution': self.df['category'].value_counts().to_dict(),
            'vocabulary_size': len(self.vocabulary.word_to_idx),
            'max_sequence_length': self.max_length,
            'preprocessing_time_seconds': preprocessing_time
        })
        
        # Store model configuration
        self.results['model_config'] = {
            'max_vocab_size': self.max_vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist(),
            'device': str(self.device)
        }
        
        print(f"Vocabulary size: {len(self.vocabulary.word_to_idx)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Final dataset size: {len(self.df)} samples")
        print(f"Preprocessing time: {preprocessing_time:.2f} seconds")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.padded_sequences,
            self.categorical_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.categorical_labels
        )
        
        # Update results with split information
        self.results['dataset_info'].update({
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'test_split_ratio': test_size,
            'random_state': random_state
        })
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def create_model(self):
        """Create the BiLSTM model"""
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocabulary.word_to_idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            max_length=self.max_length
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Update results with model info
        self.results['model_config'].update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss'
        })
        
        print(f"Model created with {total_params} parameters ({trainable_params} trainable)")
    
    def train(self, epochs=30, batch_size=32, validation_split=0.1):
        """Train the model"""
        # Record training start time
        self.results['training_start_time'] = datetime.datetime.now().isoformat()
        training_start_time = time.time()
        
        # Create validation split
        val_size = int(len(self.X_train) * validation_split)
        X_val = self.X_train[:val_size]
        y_val = self.y_train[:val_size]
        X_train_final = self.X_train[val_size:]
        y_train_final = self.y_train[val_size:]
        
        # Create datasets and data loaders
        train_dataset = NewsDataset(X_train_final, y_train_final)
        val_dataset = NewsDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Store training configuration
        self.results['model_config'].update({
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'train_final_size': len(X_train_final),
            'val_size': len(X_val)
        })
        
        print("Starting training...")
        
        early_stopping = EarlyStopping(patience=3, monitor='val_f1')
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                _, target_labels = torch.max(target, 1)
                train_total += target.size(0)
                train_correct += (predicted == target_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_predictions = []
            all_val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    _, target_labels = torch.max(target, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target_labels).sum().item()
                    
                    # Collect predictions and targets for F1 score
                    all_val_predictions.extend(predicted.cpu().numpy())
                    all_val_targets.extend(target_labels.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Calculate F1 score using all validation predictions
            val_f1 = f1_score(all_val_targets, all_val_predictions, average='weighted')
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} '
                  f'Time: {epoch_time:.2f}s')
            
            # Early stopping
            early_stopping(val_f1, self.model)
            if early_stopping.early_stop:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                print(f"Best validation F1 score: {early_stopping.best_score:.4f}")
                break
        
        # Record training completion
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        self.results['training_end_time'] = datetime.datetime.now().isoformat()
        self.results['training_duration'] = total_training_time
        
        # Store detailed training results
        self.results['training_history'] = {
            'total_epochs_trained': len(self.history['train_loss']),
            'planned_epochs': epochs,
            'early_stopped': len(self.history['train_loss']) < epochs,
            'best_val_f1': max(self.history['val_f1']),
            'best_val_acc': max(self.history['val_acc']),
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'final_val_f1': self.history['val_f1'][-1],
            'total_training_time_seconds': total_training_time,
            'total_training_time_formatted': str(datetime.timedelta(seconds=int(total_training_time))),
            'average_epoch_time_seconds': np.mean(epoch_times),
            'epoch_times_seconds': epoch_times,
            'history': self.history
        }
        
        print(f"\n✅ Training completed in {datetime.timedelta(seconds=int(total_training_time))}")
        print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
    
    def evaluate(self):
        """Evaluate the model on test set"""
        evaluation_start_time = time.time()
        
        test_dataset = NewsDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Get probabilities
                probabilities = torch.softmax(output, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                
                _, predicted = torch.max(output.data, 1)
                _, target_labels = torch.max(target, 1)
                
                test_total += target.size(0)
                test_correct += (predicted == target_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())
        
        evaluation_time = time.time() - evaluation_start_time
        
        # Calculate comprehensive metrics
        test_accuracy = test_correct / test_total
        test_f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        test_f1_macro = f1_score(all_targets, all_predictions, average='macro')
        test_f1_micro = f1_score(all_targets, all_predictions, average='micro')
        
        # Get precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average=None)
        
        # Classification report
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=self.label_encoder.classes_,
            output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Store detailed results
        self.results['test_results'] = {
            'test_accuracy': test_accuracy,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro,
            'evaluation_time_seconds': evaluation_time,
            'total_test_samples': test_total,
            'correct_predictions': test_correct,
            'incorrect_predictions': test_total - test_correct
        }
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        self.results['detailed_metrics'] = {
            'per_class_metrics': class_metrics,
            'classification_report': class_report,
            'macro_avg': {
                'precision': class_report['macro avg']['precision'],
                'recall': class_report['macro avg']['recall'],
                'f1_score': class_report['macro avg']['f1-score']
            },
            'weighted_avg': {
                'precision': class_report['weighted avg']['precision'],
                'recall': class_report['weighted avg']['recall'],
                'f1_score': class_report['weighted avg']['f1-score']
            }
        }
        
        # Store confusion matrix
        self.results['confusion_matrix'] = conf_matrix.tolist()
        
        # Store raw predictions and probabilities for further analysis
        self.results['predictions'] = {
            'predicted_labels': all_predictions,
            'true_labels': all_targets,
            'predicted_probabilities': [prob.tolist() for prob in all_probabilities]
        }
        
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Test F1 Score (Weighted): {test_f1_weighted:.4f}')
        print(f'Test F1 Score (Macro): {test_f1_macro:.4f}')
        print(f'Test F1 Score (Micro): {test_f1_micro:.4f}')
        print(f'Evaluation time: {evaluation_time:.2f} seconds')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.label_encoder.classes_))
        
        return all_predictions, all_targets
    
    def plot_training_history(self, save_plots=True, results_dir='results'):
        """Plot training history"""
        if save_plots:
            Path(results_dir).mkdir(exist_ok=True)
            
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history['val_f1'], label='Validation F1 Score', color='green')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{results_dir}/training_history.png', dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {results_dir}/training_history.png")
        
        plt.show()
    
    def plot_confusion_matrix(self, predictions, targets, save_plots=True, results_dir='results'):
        """Plot confusion matrix"""
        if save_plots:
            Path(results_dir).mkdir(exist_ok=True)
            
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_plots:
            plt.savefig(f'{results_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix plot saved to {results_dir}/confusion_matrix.png")
        
        plt.show()
    
    def plot_class_performance(self, save_plots=True, results_dir='results'):
        """Plot per-class performance metrics"""
        if save_plots:
            Path(results_dir).mkdir(exist_ok=True)
            
        classes = list(self.results['detailed_metrics']['per_class_metrics'].keys())
        precision = [self.results['detailed_metrics']['per_class_metrics'][c]['precision'] for c in classes]
        recall = [self.results['detailed_metrics']['per_class_metrics'][c]['recall'] for c in classes]
        f1 = [self.results['detailed_metrics']['per_class_metrics'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{results_dir}/class_performance.png', dpi=300, bbox_inches='tight')
            print(f"📊 Class performance plot saved to {results_dir}/class_performance.png")
        
        plt.show()
    
    def predict(self, text):
        """Predict category for a single text"""
        self.model.eval()
        
        # Preprocess text
        normalized = self.preprocessor.normalize_text(text)
        lemmatized = self.preprocessor.lemmatize_text(normalized)
        
        # Convert to sequence
        sequence = self.vocabulary.texts_to_sequences([lemmatized])
        padded = pad_sequences(sequence, max_length=self.max_length)
        
        # Convert to tensor
        input_tensor = torch.LongTensor(padded).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        predicted_class = self.label_encoder.inverse_transform([predicted.cpu().item()])[0]
        confidence = probabilities.max().cpu().item()
        
        return predicted_class, confidence
    
    def save_model(self, model_path='turkish_news_model.pth', vocab_path='vocabulary.pkl'):
        """Save model and vocabulary"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_,
            'max_vocab_size': self.max_vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes
        }, model_path)
        
        self.vocabulary.save(vocab_path)
        print(f"Model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
    
    def load_model(self, model_path='turkish_news_model.pth', vocab_path='vocabulary.pkl'):
        """Load model and vocabulary"""
        try:
            # Try loading with weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            # Fallback for older PyTorch versions or other issues
            print(f"Warning: Failed to load with weights_only=False, trying default method: {str(e)}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e2:
                raise Exception(f"Failed to load model: {str(e2)}")
        
        # Restore parameters
        self.max_vocab_size = checkpoint['max_vocab_size']
        self.max_length = checkpoint['max_length']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.dropout_rate = checkpoint['dropout_rate']
        self.num_classes = checkpoint['num_classes']
        
        # Load vocabulary
        self.vocabulary.load(vocab_path)
        
        # Create and load model
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocabulary.word_to_idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            max_length=self.max_length
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = checkpoint['label_encoder_classes']
        
        print(f"Model loaded from {model_path}")

    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        else:
            return obj
    
    def save_results(self, results_dir='results'):
        """Save comprehensive results to files"""
        # Create results directory
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results as JSON
        results_file = results_path / f'training_results_{timestamp}.json'
        
        try:
            # Convert data to JSON-serializable format
            json_safe_results = self._convert_for_json(self.results)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"✅ Results JSON saved to {results_file}")
        except Exception as e:
            print(f"❌ Error saving JSON results: {str(e)}")
            print("Attempting to save without problematic data...")
            # Create a minimal version without potentially problematic data
            minimal_results = {
                'dataset_info': self.results.get('dataset_info', {}),
                'model_config': self.results.get('model_config', {}),
                'training_history': {
                    'total_epochs_trained': self.results.get('training_history', {}).get('total_epochs_trained', 0),
                    'total_training_time_formatted': self.results.get('training_history', {}).get('total_training_time_formatted', ''),
                    'best_val_f1': self.results.get('training_history', {}).get('best_val_f1', 0),
                },
                'test_results': self.results.get('test_results', {}),
            }
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                print(f"✅ Minimal results JSON saved to {results_file}")
            except Exception as e2:
                print(f"❌ Failed to save even minimal JSON: {str(e2)}")
                print("Continuing without JSON save...")
        
        # Save training history as CSV
        history_df = pd.DataFrame(self.history)
        history_df.index.name = 'epoch'
        history_file = results_path / f'training_history_{timestamp}.csv'
        history_df.to_csv(history_file)
        
        # Save detailed metrics as CSV
        metrics_data = []
        for class_name, metrics in self.results['detailed_metrics']['per_class_metrics'].items():
            metrics_data.append({
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'support': metrics['support']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = results_path / f'class_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save confusion matrix as CSV
        conf_matrix_df = pd.DataFrame(
            self.results['confusion_matrix'],
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        conf_matrix_file = results_path / f'confusion_matrix_{timestamp}.csv'
        conf_matrix_df.to_csv(conf_matrix_file)
        
        # Generate and save comprehensive text report
        self.generate_text_report(results_path, timestamp)
        
        return results_path, timestamp
    
    def generate_text_report(self, results_path, timestamp):
        """Generate a comprehensive text report"""
        report_file = results_path / f'comprehensive_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TURKISH NEWS CLASSIFICATION - COMPREHENSIVE RESULTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset Information
            f.write("📊 DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            dataset_info = self.results['dataset_info']
            f.write(f"Dataset file: {dataset_info['csv_path']}\n")
            f.write(f"Original samples: {dataset_info['original_size']:,}\n")
            f.write(f"Final samples (after augmentation): {dataset_info['final_size']:,}\n")
            f.write(f"Augmentation factor: {dataset_info.get('augmentation_factor', 'N/A')}\n")
            f.write(f"Training samples: {dataset_info['train_size']:,}\n")
            f.write(f"Test samples: {dataset_info['test_size']:,}\n")
            f.write(f"Classes: {len(dataset_info['categories'])}\n")
            f.write(f"Class names: {', '.join(dataset_info['categories'])}\n")
            f.write(f"Vocabulary size: {dataset_info['vocabulary_size']:,}\n")
            f.write(f"Max sequence length: {dataset_info['max_sequence_length']}\n")
            f.write(f"Preprocessing time: {dataset_info['preprocessing_time_seconds']:.2f} seconds\n\n")
            
            # Model Configuration
            f.write("🧠 MODEL CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            model_config = self.results['model_config']
            f.write(f"Model type: Bidirectional LSTM\n")
            f.write(f"Embedding dimension: {model_config['embedding_dim']}\n")
            f.write(f"Hidden dimension: {model_config['hidden_dim']}\n")
            f.write(f"Dropout rate: {model_config['dropout_rate']}\n")
            f.write(f"Learning rate: {model_config['learning_rate']}\n")
            f.write(f"Total parameters: {model_config['total_parameters']:,}\n")
            f.write(f"Trainable parameters: {model_config['trainable_parameters']:,}\n")
            f.write(f"Optimizer: {model_config['optimizer']}\n")
            f.write(f"Loss function: {model_config['loss_function']}\n")
            f.write(f"Device: {model_config['device']}\n\n")
            
            # Training Information
            f.write("🏋️ TRAINING INFORMATION\n")
            f.write("-" * 40 + "\n")
            training_info = self.results['training_history']
            f.write(f"Training start: {self.results['training_start_time']}\n")
            f.write(f"Training end: {self.results['training_end_time']}\n")
            f.write(f"Total training time: {training_info['total_training_time_formatted']}\n")
            f.write(f"Planned epochs: {training_info['planned_epochs']}\n")
            f.write(f"Actual epochs trained: {training_info['total_epochs_trained']}\n")
            f.write(f"Early stopping: {'Yes' if training_info['early_stopped'] else 'No'}\n")
            f.write(f"Average time per epoch: {training_info['average_epoch_time_seconds']:.2f} seconds\n")
            f.write(f"Batch size: {model_config['batch_size']}\n\n")
            
            # Training Results
            f.write("📈 TRAINING RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best validation F1 score: {training_info['best_val_f1']:.4f}\n")
            f.write(f"Best validation accuracy: {training_info['best_val_acc']:.4f}\n")
            f.write(f"Final training loss: {training_info['final_train_loss']:.4f}\n")
            f.write(f"Final training accuracy: {training_info['final_train_acc']:.4f}\n")
            f.write(f"Final validation loss: {training_info['final_val_loss']:.4f}\n")
            f.write(f"Final validation accuracy: {training_info['final_val_acc']:.4f}\n")
            f.write(f"Final validation F1: {training_info['final_val_f1']:.4f}\n\n")
            
            # Test Results
            f.write("🧪 TEST RESULTS\n")
            f.write("-" * 40 + "\n")
            test_results = self.results['test_results']
            f.write(f"Test accuracy: {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)\n")
            f.write(f"Test F1-score (weighted): {test_results['test_f1_weighted']:.4f}\n")
            f.write(f"Test F1-score (macro): {test_results['test_f1_macro']:.4f}\n")
            f.write(f"Test F1-score (micro): {test_results['test_f1_micro']:.4f}\n")
            f.write(f"Correct predictions: {test_results['correct_predictions']:,}\n")
            f.write(f"Incorrect predictions: {test_results['incorrect_predictions']:,}\n")
            f.write(f"Evaluation time: {test_results['evaluation_time_seconds']:.2f} seconds\n\n")
            
            # Overall Performance Summary
            f.write("📋 OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            detailed_metrics = self.results['detailed_metrics']
            f.write(f"Macro average precision: {detailed_metrics['macro_avg']['precision']:.4f}\n")
            f.write(f"Macro average recall: {detailed_metrics['macro_avg']['recall']:.4f}\n")
            f.write(f"Macro average F1-score: {detailed_metrics['macro_avg']['f1_score']:.4f}\n")
            f.write(f"Weighted average precision: {detailed_metrics['weighted_avg']['precision']:.4f}\n")
            f.write(f"Weighted average recall: {detailed_metrics['weighted_avg']['recall']:.4f}\n")
            f.write(f"Weighted average F1-score: {detailed_metrics['weighted_avg']['f1_score']:.4f}\n\n")
            
            # Per-class Performance
            f.write("📊 PER-CLASS PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}\n")
            f.write("-" * 55 + "\n")
            
            for class_name, metrics in detailed_metrics['per_class_metrics'].items():
                f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1_score']:<10.4f} {metrics['support']:<8}\n")
            
            f.write("\n")
            
            # Training History Summary
            f.write("📉 TRAINING HISTORY SUMMARY\n")
            f.write("-" * 40 + "\n")
            history = training_info['history']
            f.write(f"Epoch-by-epoch performance:\n")
            f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12} {'Time(s)':<8}\n")
            f.write("-" * 80 + "\n")
            
            for i in range(len(history['train_loss'])):
                epoch_time = training_info['epoch_times_seconds'][i] if i < len(training_info['epoch_times_seconds']) else 0
                f.write(f"{i+1:<6} {history['train_loss'][i]:<12.4f} {history['train_acc'][i]:<12.4f} "
                       f"{history['val_loss'][i]:<12.4f} {history['val_acc'][i]:<12.4f} "
                       f"{history['val_f1'][i]:<12.4f} {epoch_time:<8.2f}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("Report generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")
        
        print(f"📋 Comprehensive report saved to {report_file}")
    
    def save_all_results_and_plots(self, results_dir='results'):
        """Save everything: results, plots, and generate comprehensive report"""
        print(f"\n💾 Saving comprehensive results to '{results_dir}' directory...")
        
        # Save structured results
        results_path, timestamp = self.save_results(results_dir)
        
        # Save all plots
        self.plot_training_history(save_plots=True, results_dir=results_dir)
        
        # Only plot confusion matrix and class performance if we have evaluation results
        if 'test_results' in self.results and self.results['test_results']:
            predictions = self.results['predictions']['predicted_labels']
            targets = self.results['predictions']['true_labels']
            self.plot_confusion_matrix(predictions, targets, save_plots=True, results_dir=results_dir)
            self.plot_class_performance(save_plots=True, results_dir=results_dir)
        
        print(f"\n✅ All results saved successfully!")
        print(f"📁 Results directory: {results_path}")
        print(f"📄 Files saved:")
        print(f"   - training_results_{timestamp}.json (Complete results)")
        print(f"   - training_history_{timestamp}.csv (Training metrics)")
        print(f"   - class_metrics_{timestamp}.csv (Per-class performance)")
        print(f"   - confusion_matrix_{timestamp}.csv (Confusion matrix)")
        print(f"   - comprehensive_report_{timestamp}.txt (Human-readable report)")
        print(f"   - training_history.png (Training plots)")
        if 'test_results' in self.results and self.results['test_results']:
            print(f"   - confusion_matrix.png (Confusion matrix plot)")
            print(f"   - class_performance.png (Per-class performance plot)")
        
        return results_path


def main():
    """Main function to run the Turkish news classification"""
    # Initialize classifier with data augmentation
    classifier = TurkishNewsClassifier(
        embedding_dim=1024, 
        hidden_dim=1024,
        use_augmentation=True,      # Enable data augmentation
        augmentation_factor=3.0   # 1.5x more data through augmentation
    )
    
    # Load data (you need to provide the path to your CSV file)
    csv_path = "data/datas.csv"  # Update this path
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        print("Please make sure you have the Turkish news dataset CSV file.")
        return
    
    try:
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        classifier.load_data(csv_path)
        classifier.preprocess_data()
        classifier.split_data()
        
        # Create and train model
        print("🧠 Creating and training model...")
        classifier.create_model()
        classifier.train(epochs=30, batch_size=32)
        
        # Evaluate model
        print("📈 Evaluating model...")
        predictions, targets = classifier.evaluate()
        
        # Save comprehensive results and plots
        print("💾 Saving comprehensive results and plots...")
        results_path = classifier.save_all_results_and_plots()
        
        # Save model with explicit confirmation
        print("\n💾 Saving trained model...")
        try:
            classifier.save_model('turkish_news_model.pth', 'vocabulary.pkl')
            print("✅ Model and vocabulary saved successfully!")
            print("   - Model file: turkish_news_model.pth")
            print("   - Vocabulary file: vocabulary.pkl")
            print("   - You can now use these files with test_model.py")
        except Exception as save_error:
            print(f"❌ Error saving model: {str(save_error)}")
            # Try alternative save paths
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                alt_model_path = f'model_backup_{timestamp}.pth'
                alt_vocab_path = f'vocab_backup_{timestamp}.pkl'
                classifier.save_model(alt_model_path, alt_vocab_path)
                print(f"✅ Model saved with backup names:")
                print(f"   - Model file: {alt_model_path}")
                print(f"   - Vocabulary file: {alt_vocab_path}")
            except Exception as backup_error:
                print(f"❌ Failed to save backup as well: {str(backup_error)}")
        
        # Example prediction
        print("\n🧪 Testing with example prediction...")
        sample_text = "Türkiye ekonomisi büyümeye devam ediyor."
        try:
            predicted_class, confidence = classifier.predict(sample_text)
            print(f"📝 Example prediction:")
            print(f"   Text: '{sample_text}'")
            print(f"   Predicted category: {predicted_class}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        except Exception as pred_error:
            print(f"❌ Error in example prediction: {str(pred_error)}")
        
        # Display comprehensive results summary
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 FINAL RESULTS SUMMARY:")
        print(f"   - Test Accuracy: {classifier.results['test_results']['test_accuracy']:.4f} ({classifier.results['test_results']['test_accuracy']*100:.2f}%)")
        print(f"   - Test F1-Score (Weighted): {classifier.results['test_results']['test_f1_weighted']:.4f}")
        print(f"   - Training Time: {classifier.results['training_history']['total_training_time_formatted']}")
        print(f"   - Epochs Trained: {classifier.results['training_history']['total_epochs_trained']}")
        print(f"   - Model Parameters: {classifier.results['model_config']['total_parameters']:,}")
        print(f"   - Final Dataset Size: {classifier.results['dataset_info']['final_size']:,} samples")
        print(f"\n📁 Results saved to: {results_path}")
        print("You can now test the model by running: python test_model.py")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the dataset file '{csv_path}'")
        print("Please make sure the Turkish news dataset is available.")
    except Exception as e:
        print(f"❌ An error occurred during training: {str(e)}")
        print("Training failed. Please check your data and try again.")


if __name__ == "__main__":
    main() 