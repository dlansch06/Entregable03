

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


def prepare_data():
    """Carga, normaliza y remodela los datos para los tres modelos."""
    print("--- 1. Carga y Preprocesamiento de Datos ---")

    (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

    X_train_norm = X_train_raw.astype('float32') / 255.0
    X_test_norm = X_test_raw.astype('float32') / 255.0

    X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1) 
    X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)   

    X_train_cnn = X_train_norm.reshape((*X_train_norm.shape, 1))
    X_test_cnn = X_test_norm.reshape((*X_test_norm.shape, 1))
    y_train_cnn = to_categorical(y_train, 10)
    y_test_cnn = to_categorical(y_test, 10)
    
    print("Datos preprocesados correctamente.")
    return X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, \
           X_train_flat, y_train, X_test_flat, y_test, X_test_raw

def build_and_train_cnn(X_train, y_train):
    """Crea la arquitectura simple de CNN y la entrena."""
    print("\n--- 2. Entrenamiento de CNN ---")
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0) 
    print("CNN entrenada.")
    return model

def train_classical_models(X_train_flat, y_train):
    """Entrena los modelos KNN y SVM (utilizando un subconjunto para SVM)."""
    print("\n--- 3. Entrenamiento de KNN y SVM ---")

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_flat, y_train)
    print("KNN entrenado.")

    sample_size = 10000 
    svm_model = SVC(kernel='rbf', gamma='scale', verbose=False)
    svm_model.fit(X_train_flat[:sample_size], y_train[:sample_size]) 
    print(f"SVM entrenado (con {sample_size} muestras).")

    return knn_model, svm_model
def evaluate_and_report(cnn_model, knn_model, svm_model, X_test_cnn, y_test_cnn, X_test_flat, y_test):
    """Evalúa los tres modelos y muestra el resumen de métricas."""
    print("\n--- 4. Evaluación de Modelos y Reporte ---")
   
    loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)

    knn_accuracy = knn_model.score(X_test_flat, y_test)
    y_pred_knn = knn_model.predict(X_test_flat)
    svm_accuracy = svm_model.score(X_test_flat, y_test)

    
    print("|              RESUMEN DE PRECISIÓN                |")
    print(f"| CNN: {cnn_accuracy*100:.2f}% |")
    print(f"| KNN: {knn_accuracy*100:.2f}% |")
    print(f"| SVM: {svm_accuracy*100:.2f}% |")
    
    print("\n--- Métricas Detalladas (CNN) ---")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_cnn))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_cnn))
    
    return y_pred_cnn

def visualize(X_test_raw, y_test, y_pred):
    """Muestra una comparación visual de las primeras 10 predicciones de la CNN."""
    print("\n--- 5. Resultados Visuales (CNN) ---")
    fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 4))
    
    for i in range(10):
        axes[i].imshow(X_test_raw[i], cmap='gray')
        is_correct = (y_test[i] == y_pred[i])
        color = 'green' if is_correct else 'red'
        
        axes[i].set_title(f"R:{y_test[i]}/P:{y_pred[i]}", color=color, fontsize=10)
        axes[i].axis('off')

    plt.suptitle('R: Real | P: Predicho', fontsize=12)
    plt.show()

if __name__ == "__main__":
    
    
    X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, \
    X_train_flat, y_train, X_test_flat, y_test, X_test_raw = prepare_data()
    cnn_model = build_and_train_cnn(X_train_cnn, y_train_cnn)
    knn_model, svm_model = train_classical_models(X_train_flat, y_train)

    y_pred_cnn = evaluate_and_report(cnn_model, knn_model, svm_model, 
                                     X_test_cnn, y_test_cnn, X_test_flat, y_test)
     
    visualize(X_test_raw, y_test, y_pred_cnn)