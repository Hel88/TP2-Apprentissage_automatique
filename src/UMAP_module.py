import umap
from matplotlib import pyplot as plt


class UMAP_module:
    def __init__(self, X, y):
        """
        Initialisation
        """
        self.embedding = None
        self.X = X
        self.y = y

    def run_algo(self):
        print("UMAP algo ...")
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(self.X)
        self.embedding = embedding
        print("done")
        return embedding

    def display(self, plot_title, file_title):
        if self.embedding is None:
            print("Run the umap algorithm first")
            return

        print("Plot umap...")
        plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.y.Cover_Type, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar()
        plt.title(plot_title)

        # save the plot
        plt.savefig("../images/"+file_title+".png")
        print("plot saved to ../images/"+ file_title+".png")
        return
