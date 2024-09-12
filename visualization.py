#!/usr/bin/env python3

__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Majumdar, Subho"]
__email__ = "md.ahsanayub@outlook.com"
__status__ = "Prototype"

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# Driver program
if __name__ == '__main__':

    df = pd.read_pickle("dataset/openai_X_train.pkl")
    df["class"] = pd.read_pickle("dataset/openai_Y_train.pkl").tolist()
    print(df.head())
    print(df.shape)

    df = df.groupby('class', group_keys=False).apply(lambda x: x.sample(1000))
    print(df.head())
    print(df.shape)

    Y = df.iloc[:, -1].values
    df = df.drop(columns=["class"])
    X = df.iloc[:, :].values

    reducer = umap.UMAP()
    df = reducer.fit_transform(X)
    UmapEmbedding = pd.DataFrame(data = df, columns = ["Component1", "Component2"])
    UmapEmbedding["class"] = Y
    print(UmapEmbedding.head())
    print(UmapEmbedding.shape)

    # 2D Visualization
    plt.clf() # Clear figure
    myFig = plt.figure(figsize=[12,12])
    plt.scatter(UmapEmbedding["Component1"][UmapEmbedding["class"] == 0],
                UmapEmbedding["Component2"][UmapEmbedding["class"] == 0],
                marker='o', alpha=0.7, color='blue')
    plt.scatter(UmapEmbedding["Component1"][UmapEmbedding["class"] == 1],
                UmapEmbedding["Component2"][UmapEmbedding["class"] == 1],
                marker='x', alpha=0.7, color='red')
    plt.title("UMAP on OpenAI Embedding", fontsize=20, weight='bold')
    plt.xlabel('Component 1', fontsize=18, weight='bold')
    plt.ylabel('Component 2', fontsize=18, weight='bold')
    plt.yticks(fontsize=16)
    plt.legend(['Benign', 'Malicious'], fontsize=16, loc='best')
    myFig.savefig("./results/graphs/umap_on_openai.png", dpi = 150, format = 'png')
    myFig.savefig("./results/graphs/umap_on_openai.pdf", dpi = 300, format = 'pdf')

    tsne = TSNE(n_components=2, learning_rate='auto', init='random',  perplexity=15)
    df = tsne.fit_transform(X)
    TsneEmbedding = pd.DataFrame(data = df, columns = ["Component1", "Component2"])
    TsneEmbedding["class"] = Y
    print(TsneEmbedding.head())
    print(TsneEmbedding.shape)

    # 2D Visualization
    plt.clf() # Clear figure
    myFig = plt.figure(figsize=[12,12])
    plt.scatter(TsneEmbedding["Component1"][TsneEmbedding["class"] == 0],
                TsneEmbedding["Component2"][TsneEmbedding["class"] == 0],
                marker='o', alpha=0.7, color='blue')
    plt.scatter(TsneEmbedding["Component1"][TsneEmbedding["class"] == 1],
                TsneEmbedding["Component2"][TsneEmbedding["class"] == 1],
                marker='x', alpha=0.7, color='red')
    plt.title("t-SNE on OpenAI Embedding", fontsize=20, weight='bold')
    plt.xlabel('Component 1', fontsize=18, weight='bold')
    plt.ylabel('Component 2', fontsize=18, weight='bold')
    plt.yticks(fontsize=16)
    plt.legend(['Benign', 'Malicious'], fontsize=16, loc='best')
    myFig.savefig("./results/graphs/tsne_on_openai.png", dpi = 150, format = 'png')
    myFig.savefig("./results/graphs/tsne_on_openai.pdf", dpi = 300, format = 'pdf')

    pca = PCA(n_components = 2)
    df = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    PrincipalComponents = pd.DataFrame(data = df, columns = ["PCA-1", "PCA-2"])
    PrincipalComponents["class"] = Y
    print(PrincipalComponents.head())
    print(PrincipalComponents.shape)

    # 2D Visualization
    plt.clf() # Clear figure
    myFig = plt.figure(figsize=[12,12])
    plt.scatter(PrincipalComponents["PCA-1"][PrincipalComponents["class"] == 0],
                PrincipalComponents["PCA-2"][PrincipalComponents["class"] == 0],
                marker='o', alpha=0.7, color='blue')
    plt.scatter(PrincipalComponents["PCA-1"][PrincipalComponents["class"] == 1],
                PrincipalComponents["PCA-2"][PrincipalComponents["class"] == 1],
                marker='x', alpha=0.7, color='red')
    plt.title("PCA on OpenAI Embedding", fontsize=20, weight='bold')
    plt.xlabel('PCA-1', fontsize=18, weight='bold')
    plt.ylabel('PCA-2', fontsize=18, weight='bold')
    plt.yticks(fontsize=16)
    plt.legend(['Benign', 'Malicious'], fontsize=16, loc='best')
    myFig.savefig("./results/graphs/pca_on_openai.png", dpi = 150, format = 'png')
    myFig.savefig("./results/graphs/pca_on_openai.pdf", dpi = 300, format = 'pdf')