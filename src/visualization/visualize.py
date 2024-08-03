import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the images directory exists
os.makedirs('visualization/images', exist_ok=True)

def plot_scatter(df):
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=df, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance')
    plt.savefig('src/visualization/images/scatterplot.png')
    plt.show()

def plot_distributions(data, Xtrain):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data['GRE_Score'], kde=True)
    plt.subplot(2, 2, 2)
    sns.histplot(Xtrain[:, 0], kde=True)
    plt.subplot(2, 2, 3)
    sns.histplot(data['TOEFL_Score'], kde=True)
    plt.subplot(2, 2, 4)
    sns.histplot(Xtrain[:, 1], kde=True)
    plt.savefig('src/visualization/images/distributions.png')
    plt.show()

def plot_loss_curve(loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('src/visualization/images/loss_curve.png')
    plt.show()
