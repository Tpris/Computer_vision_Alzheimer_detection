import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from src.arrange_dataset import arrange_dataset
from src.data_loader3D import dataGenerator, NiiSequence
from sklearn.manifold import TSNE
from keras.models import Model

if __name__ == "__main__":
    print("Loading model")
    model = load_model("experiences_2/classifier3D_bi-exp2-07-0.46")
    print("Model loaded")

    print("Loading data")
    data_dir = arrange_dataset()
    batch_size = 16
    # set_SP, labels_SP = dataGenerator(data_dir, mode="val", nb_classes=4)
    # test_sequence = NiiSequence(set_SP, batch_size, nb_classes=4, mode="HC", shuffle=False)

    set_train, labels_train = dataGenerator(data_dir, mode="train", nb_classes=4)
    batch_size = 16
    train_sequence = NiiSequence(set_train, batch_size, nb_classes=4, mode="HC", shuffle=True)

    num_samples_for_visualization = 200  # Nombre d'échantillons pour la visualisation
    X_visualize = []
    y_visualize = []
    i = 0
    while len(X_visualize) < num_samples_for_visualization and i < len(train_sequence):
        X, y = train_sequence[i]
        X_visualize.extend(X)
        y_visualize.extend(y)
        i += 1
    X_visualize = np.array(X_visualize)
    y_visualize = np.array(y_visualize)
    print(X_visualize.shape)
    y_visualize = y_visualize.argmax(axis=1)
    print(y_visualize.shape)
    layer_output_model = model.layers[-1].output  # Récupérer la sortie de la dernière couche du modèle
    feature_extraction_model = Model(inputs=model.input, outputs=layer_output_model)
    features = feature_extraction_model.predict(X_visualize)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)

    # Visualiser les données
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_visualize, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Visualization of Brain MRI Data')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()