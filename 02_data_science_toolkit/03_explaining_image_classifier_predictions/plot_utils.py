import matplotlib.pyplot as plt

class PlotUtil:

    @staticmethod
    def plot_grid(images, img_shape, grid_size, cls_true=None, cls_pred=None, save_path=None):
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
        fig.subplots_adjust(hspace= 0.9 / grid_size[0], wspace= 0.9 / grid_size[1])

        for i, ax in enumerate(axes.flat):
            normalized_image = (images[i] - images[i].min())/(images[i].max() - images[i].min())
            ax.imshow(normalized_image.reshape(img_shape), interpolation='nearest')
            
            titles = []
            if cls_true is not None:
                titles.append("True: {0}".format(cls_true[i]))
            if cls_pred is not None:
                titles.append("Pred: {0}".format(cls_pred[i]))

            ax.set_title(" ".join(titles))
            ax.axis('off')
            
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.show()

    @staticmethod
    def plot_3x3_grid(images, img_shape, cls_true=None, cls_pred=None, save_path=None):
        PlotUtil.plot_grid(images, img_shape, (3, 3), cls_true=cls_true, cls_pred=cls_pred, save_path=save_path)
        
    @staticmethod
    def plot_5x5_grid(images, img_shape, cls_true=None, cls_pred=None, save_path=None):
        PlotUtil.plot_grid(images, img_shape, (5, 5), cls_true=cls_true, cls_pred=cls_pred, save_path=save_path)
        
    @staticmethod
    def plot_7x7_grid(images, img_shape, cls_true=None, cls_pred=None, save_path=None):
        PlotUtil.plot_grid(images, img_shape, (7, 7), cls_true=cls_true, cls_pred=cls_pred, save_path=save_path)