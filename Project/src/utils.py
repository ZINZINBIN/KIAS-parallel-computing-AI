import matplotlib.pyplot as plt

def plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, save_dir = "./image/learning_curve.png"):
    x_epochs = range(1, len(train_loss) + 1)

    plt.figure(1, figsize = (12,6))
    plt.subplot(1,2,1)
    plt.plot(x_epochs, train_loss, 'ro-', label = "train loss")
    plt.plot(x_epochs, valid_loss, 'bo-', label = "valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train and valid loss curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_epochs, train_f1, 'ro-', label = "train f1 score")
    plt.plot(x_epochs, valid_f1, 'bo-', label = "valid f1 score")
    plt.xlabel("epochs")
    plt.ylabel("f1 score")
    plt.title("train and valid f1 score curve")
    plt.legend()
    plt.savefig(save_dir)