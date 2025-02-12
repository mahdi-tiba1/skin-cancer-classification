import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans

class SkinCancerClassifier:
    def __init__(self, data_path, image_size=(128, 128)):
        self.data_path = data_path
        self.image_size = image_size
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_and_preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_features(self, image, feature_types):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        features = []

        if 'haralick' in feature_types:
            try:
                import mahotas
                haralick = mahotas.features.haralick(
                    gray,
                    distance=2,          # Set the distance for GLCM
                    ignore_zeros=True,   # Ignore zero-value pixels
                ).mean(axis=0)
                features.extend(haralick)
            except ImportError:
                print("Warning: Mahotas library is not installed. Skipping Haralick features.")

        if 'lbp' in feature_types:
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
            lbp_features = lbp_hist.astype("float") / lbp_hist.sum()
            features.extend(lbp_features)

        if 'tamura' in feature_types:
            features.extend(self.extract_tamura_features(gray))

        if 'markovian' in feature_types:
            features.extend(self.extract_markovian_features(gray))

        if 'hog' in feature_types:
            from skimage.feature import hog
            hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            features.extend(hog_features)

        if 'wld' in feature_types:
            features.extend(self.extract_wld_features(gray))

        if 'run' in feature_types:
            features.extend(self.extract_run_length_features(gray))

        return np.array(features)

    def extract_tamura_features(self, gray):
        scales = [2, 4, 8, 16]
        height, width = gray.shape
        coarseness_values = []
        for scale in scales:
            kernel = np.ones((scale, scale)) / (scale ** 2)
            mean = cv2.filter2D(gray, -1, kernel)
            variance = cv2.filter2D(gray ** 2, -1, kernel) - mean ** 2
            coarseness_values.append(variance.mean())
        coarseness = np.mean(coarseness_values)

        patch_size = 16
        patches = [
            gray[i:i+patch_size, j:j+patch_size]
            for i in range(0, height, patch_size)
            for j in range(0, width, patch_size)
        ]
        local_contrast = [patch.std() for patch in patches if patch.size > 0]
        contrast_mean = np.mean(local_contrast)
        contrast_std = np.std(local_contrast)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_directions = np.arctan2(sobely, sobelx)
        bins = 72
        hist, _ = np.histogram(gradient_directions, bins=bins, range=(-np.pi, np.pi), weights=gradient_magnitude)
        directionality_entropy = entropy(hist)
        dominant_direction = np.argmax(hist)

        return np.array([coarseness, contrast_mean, contrast_std, directionality_entropy, dominant_direction])

    def extract_markovian_features(self, gray):
        quantized = (gray / 16).astype(int)
        num_levels = 16
        transitions = np.zeros((num_levels, num_levels))

        for i in range(quantized.shape[0] - 1):
            for j in range(quantized.shape[1] - 1):
                curr = quantized[i, j]
                neighbors = [
                    quantized[i + 1, j],      
                    quantized[i, j + 1],      
                    quantized[i + 1, j + 1],  
                    quantized[i + 1, j - 1],  
                ]
                for n in neighbors:
                    transitions[curr, n] += abs(curr - n) 

        row_sums = transitions.sum(axis=1, keepdims=True)
        markov_matrix = transitions / (row_sums + 1e-6)

        matrix_mean = markov_matrix.mean()
        matrix_entropy = entropy(markov_matrix.flatten())
        matrix_variance = markov_matrix.var()

        return np.concatenate([markov_matrix.flatten(), [matrix_mean, matrix_entropy, matrix_variance]])

    def extract_wld_features(self, gray):
        gray_blurred = gaussian_filter(gray, sigma=1)
        dx = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        gradient_orientation = np.arctan2(dy, dx)
        mean_orientation = gradient_orientation.mean()
        contrast = gradient_magnitude.var()
        return [mean_orientation, contrast]

    def extract_run_length_features(self, gray):
        levels = 8
        quantized = (gray / (256 // levels)).astype(np.uint8)
        glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(graycoprops(glcm, prop)[0, 0])
        return features

    def load_dataset(self, feature_types):
        features_list = []
        labels_list = []

        for split in ['train', 'test']:
            split_path = os.path.join(self.data_path, split)
            for label, class_name in enumerate(['benign', 'malignant']):
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: Directory {class_path} does not exist!")
                    continue

                for img_name in tqdm(os.listdir(class_path), desc=f'Processing {split}/{class_name}'):
                    img_path = os.path.join(class_path, img_name)
                    image = self.load_and_preprocess_image(img_path)
                    if image is not None:
                        features = self.extract_features(image, feature_types)
                        features_list.append(features)
                        labels_list.append(label)

        return np.array(features_list), np.array(labels_list)

    def train_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def train_kmeans(self, X_train, n_clusters=2):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(X_train)

    def evaluate(self, X_test, y_test, method='supervised'):
        if method == 'supervised':
            predictions = self.classifier.predict(X_test)
        else:
            predictions = self.kmeans.predict(X_test)
            if accuracy_score(y_test, predictions) < 0.5:
                predictions = 1 - predictions

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
        conf_matrix = confusion_matrix(y_test, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }

    def plot_results(self, metrics, method, split_info='', save_path=None):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        values = [metrics[m] for m in metrics_to_plot]
        title = f'{method.capitalize()} Learning Metrics'
        if split_info:
            title += f'\n{split_info}'
        plt.bar(metrics_to_plot, values)
        plt.title(title)
        plt.ylim(0, 1)

        plt.subplot(1, 2, 2)
        conf_matrix = metrics['confusion_matrix']
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Benign', 'Malignant'])
        plt.yticks(tick_marks, ['Benign', 'Malignant'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def main():
    data_path = r"C:\Users\Hp\Desktop\ctu courses\pattern recognition\semestral proj\pattern recognition data\archive"
    classifier = SkinCancerClassifier(data_path)

    # Define the possible feature types
    feature_types = ['haralick', 'lbp', 'tamura', 'markovian', 'hog', 'wld', 'run']

    # Get all feature combinations (1 feature to all features)
    all_combinations = []
    for r in range(1, len(feature_types) + 1):
        all_combinations.extend(combinations(feature_types, r))

    # Variables to track the best feature combination
    best_metrics = None
    best_feature_set = None

    # Loop through all combinations of features and evaluate each one
    for feature_set in all_combinations:
        print(f"Evaluating feature set: {feature_set}")
        # Load dataset with the current feature set
        features, labels = classifier.load_dataset(feature_set)
        
        # Train classifier and evaluate performance
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        classifier.train_classifier(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test, method='supervised')

        # If this combination is the best so far based on F1 score, update best_metrics
        if best_metrics is None or metrics['f1'] > best_metrics['f1']:
            best_metrics = metrics
            best_feature_set = feature_set

    # Print the best feature combination and its metrics
    print("\nBest Feature Combination:")
    print(f"Feature Set: {best_feature_set}")
    print(f"Metrics: {best_metrics}")

    # Create figures directory
    figure_dir = os.path.join(data_path, "figures_best_combination")
    os.makedirs(figure_dir, exist_ok=True)

    # Now, evaluate and generate figures for the best feature set
    features, labels = classifier.load_dataset(best_feature_set)

    # Supervised learning with different splits
    split_ratios = [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]
    for i, (train_size, test_size) in enumerate(split_ratios):
        print(f"\nTesting supervised learning with train:test split of {train_size}:{test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        classifier.train_classifier(X_train, y_train)
        supervised_metrics = classifier.evaluate(X_test, y_test, method='supervised')
        split_info = f'Train-Test Split: {train_size}:{test_size}'
        supervised_save_path = os.path.join(figure_dir, f'supervised_split_{i}.png')
        classifier.plot_results(supervised_metrics, 'supervised', split_info, supervised_save_path)

    # Single unsupervised learning run on complete dataset
    print("\nPerforming unsupervised learning on complete dataset...")
    classifier.train_kmeans(features)
    unsupervised_metrics = classifier.evaluate(features, labels, method='unsupervised')
    unsupervised_save_path = os.path.join(figure_dir, 'unsupervised_complete.png')
    classifier.plot_results(unsupervised_metrics, 'unsupervised', 'Complete Dataset', unsupervised_save_path)


if __name__ == "__main__":
    main()