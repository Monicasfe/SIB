import pandas as pd

class ConfusionMatrix:

    def __call__(self, true_y, pred_y, graph=False):
        self.true = true_y
        self.pred = pred_y
        if graph is True:
            self.to_heatmap()
            return self.to_df()
        else:
            return self.to_df()

    def calc(self):
        data = {"true_y": self.true, "pred_y": self.pred}
        df = pd.DataFrame(data, columns=["true_y", "pred_y"])
        cm = pd.crosstab(df["true_y"], df["pred_y"], rownames=["Actual Values"], colnames=["Predicted Values"], margins=True)
        return cm

    def to_df(self):
        return pd.DataFrame(self.calc())

    def to_heatmap(self):
        import seaborn as sn
        import matplotlib.pyplot as plt
        sn.heatmap(self.calc(), annot=True, cmap="Blues", fmt="g")
        plt.title("Confusion Matrix")
        plt.show()


