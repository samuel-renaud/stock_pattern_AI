import matplotlib.pyplot as plt

def loss_plot(loss_list):
    plt.figure()
    plt.title('Loss Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    return plt.plot(loss_list)