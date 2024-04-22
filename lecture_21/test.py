import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Generate synthetic circular data for binary classification
def generate_circles_data():
    X, Y = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=1)
    return X, Y


X, Y = generate_circles_data()

# Initialize polynomial features and logistic regression
poly = PolynomialFeatures(2)
lr = LogisticRegression(
    max_iter=1, solver="saga", tol=1e-2, warm_start=True, penalty="none"
)
model = make_pipeline(poly, lr)

# Prepare plot for visualization
fig, ax = plt.subplots()
points_red = ax.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color="red")
points_blue = ax.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color="blue")


# Function to update the plot
def update(i):
    model.fit(X, Y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Clear the previous contour to prevent overplotting
    for coll in ax.collections:
        coll.remove()

    contour = ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.5,
        levels=[-1, 0, 1],
        colors=["red", "blue"],
        linestyles=["--", "-"],
    )
    return (contour,)


# Creating animation
ani = FuncAnimation(fig, update, frames=np.arange(1, 50), blit=False)

plt.show()
