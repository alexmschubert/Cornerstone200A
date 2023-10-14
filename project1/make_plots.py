import json
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
    with open('results_q1.json', 'r') as file:
        results_q1 = json.load(file)
    
    #Q1 loss curve plot
    plt.plot(results_q1['train_losses'], c='b', label='Training Loss')
    plt.plot(results_q1['val_losses'], c='g', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Q1: Loss curve')
    plt.legend()
    plt.savefig('Q1_loss_curve.png')
    