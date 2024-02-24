import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels

 
def load_test_images_from_folder(folder, target_shape=None):
    test_images = []
    test_labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        test_images.append(img)
                        test_labels.append(subfolder)
                        print('Test Labels \n', test_labels)  
                    else:
                        print(f"Warning: Unable to load {filename}")
    return test_images, test_labels

def load_validation_images_from_folder(folder, target_shape=None):
    validation_images = []
    validation_labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                       
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        validation_images.append(img)
                        validation_labels.append(subfolder)
                        print('Validation Labels \n', validation_labels)  
                    else:
                        print(f"Warning: Unable to load {filename}")
    return validation_images, validation_labels


data_folder = './train' 
test_folder = './test' 
validation_folder = './val'


# Load images and labels from the 'train' folder and resize them to (100, 100)
images, labels = load_train_images_from_folder(data_folder, target_shape=(200, 200))

# Load validation images and labels from the 'val' folder
validation_images, validation_labels = load_validation_images_from_folder(validation_folder, target_shape=(200, 200))

# Combine training and validation data
images += validation_images
labels += validation_labels

# Load test images and labels from the 'test' folder
test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(200, 200))

# Convert labels to binary (0 or 1)
labels_binary = [1 if label == 'Not_fractured' else 0 for label in labels]
test_labels_binary = [1 if label == 'Not_fractured' else 0 for label in test_labels]


# Reshape the images and convert them to grayscale
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]
test_image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in test_images]

# Convert the list of 1D arrays to a 2D numpy array
image_data = np.array(image_data)
test_image_data = np.array(test_image_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)
scaled_test_data = scaler.transform(test_image_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(scaled_data, labels_binary, test_size=0.2, random_state=42)

# Train a Random Forest model
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Predictions on the validation set
validation_predictions = random_forest_model.predict(X_val)

# Evaluate performance on the validation set
accuracy_val = accuracy_score(y_val, validation_predictions)
precision_val = precision_score(y_val, validation_predictions)
recall_val = recall_score(y_val, validation_predictions, zero_division=1)
f1_val = f1_score(y_val, validation_predictions)
confusion_matrix_val = confusion_matrix(y_val, validation_predictions)

print("Performance on Validation Set:")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_val)

# Plot confusion matrix for validation set
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix_val, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Validation Set')
plt.colorbar()
classes = ['Fractured', 'Not_fractured']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#training set 
# Predictions on the training set
train_predictions = random_forest_model.predict(X_train)

# Evaluate performance on the training set
correct_predictions_train = sum(train_predictions == y_train)
total_samples_train = len(y_train)

# Calculate accuracy on the training set
accuracy_train = correct_predictions_train / total_samples_train

# Print and display the results for the training set
print("\nPerformance on Training Set:")
print(f"Accuracy: {accuracy_train:.4f}")
precision_train = precision_score(y_train, train_predictions)
recall_train = recall_score(y_train, train_predictions, zero_division=1)
f1_train = f1_score(y_train, train_predictions)
confusion_matrix_train = confusion_matrix(y_train, train_predictions)

print("Performance on Training Set:")
print(f"Accuracy: {accuracy_train:.4f}")
print(f"Precision: {precision_train:.4f}")
print(f"Recall: {recall_train:.4f}")
print(f"F1 Score: {f1_train:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_train)

# Plot confusion matrix for training set
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Training Set')
plt.colorbar()
classes = ['Fractured', 'Not_fractured']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Predictions on the test set
test_predictions = random_forest_model.predict(scaled_test_data)

# Evaluate performance on the test set
accuracy_test = accuracy_score(test_labels_binary, test_predictions)
precision_test = precision_score(test_labels_binary, test_predictions)
recall_test = recall_score(test_labels_binary, test_predictions, zero_division=1)
f1_test = f1_score(test_labels_binary, test_predictions)
confusion_matrix_test = confusion_matrix(test_labels_binary, test_predictions)

print("\nPerformance on Test Set:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_test)

# Plot confusion matrix for test set
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
classes = ['Fractured', 'Not_fractured']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

for i, test_image in enumerate(test_images):
    
    if test_predictions[i] == 1:
        print(f"Test Image {i + 1} - ==Not_fractured==")
    else:
        print(f"Test Image {i + 1} - ==Fractured==")

    
    # plt.figure()
    # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    # if test_predictions[i] == 1:
    #     plt.title(f"Test Image {i + 1} - ==Not_fractured==")
    # else:
    #     plt.title(f"Test Image {i + 1} - ==Fractured==")
    # plt.axis('off')
    # plt.show()
        
# Define seturile și acuratețea corespunzătoare
sets = ['Antrenare', 'Validare', 'Testare']  
accuracies = [0.9748, 0.6752, 0.6538]  # Poți înlocui cu acuratețile corespunzătoare fiecărui set
r
# Afișează diagrama cu acuratețea pentru fiecare set
plt.bar(sets, accuracies, color=['green', 'blue', 'orange'])
plt.xlabel('Set de Date')
plt.ylabel('Acuratețe')
plt.title('Acuratețe pe Seturile de Date')

# Adaugă numele setului de date sub fiecare bară
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, str(v), ha='center', va='bottom')

plt.ylim(0, 1)  # Asigură că axa Y începe de la 0 și se termină la 1 pentru procentaje
plt.show()

