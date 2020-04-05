import numpy as np
import matplotlib.pyplot as plt


def display_2d_data_set(x: np.array, y: np.array, output_path: str = None) -> None:
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=50,
                cmap=plt.cm.Spectral, edgecolors='white')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.show()


def display_classification_areas(
        model,
        cords: np.array,
        labels: np.array,
        output_path: str = None

) -> None:
    x_start = cords[:, 0].min() - 0.5
    x_end = cords[:, 0].max() + 0.5
    y_start = cords[:, 1].min() - 0.5
    y_end = cords[:, 1].max() + 0.5

    grid = np.mgrid[x_start:x_end:100j, y_start:y_end:100j]
    grid_2d = grid.reshape(2, -1).T
    x_grid, y_grid = grid

    prediction = model.predict(grid_2d)

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.contourf(x_grid, y_grid, prediction.reshape(100, 100),
                 alpha=0.7, cmap=plt.cm.Spectral)
    plt.scatter(cords[:, 0], cords[:, 1], c=labels.ravel(),
                s=50, cmap=plt.cm.Spectral, edgecolors='white')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.show()
