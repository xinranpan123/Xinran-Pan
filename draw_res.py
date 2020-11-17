import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

"""
"""

if False:
    W = np.loadtxt('W_matrix.txt')
    W = (W - W.min()) / (W.max() - W.min())
    X = W * 255
    X_all = []
    for i in range(8):
        X_all.append(X[i].reshape(7, 8))
        im = Image.fromarray(X_all[i])
        im = im.convert('L')
        im.save('outfile_{}.png'.format(i))
        # im.show()


if False:
    all_prediction = np.loadtxt('all_prediction.txt')
    all_y = np.loadtxt('all_y.txt')
    result_to_show = all_prediction
    # result_to_show = all_y
    count = Counter(result_to_show)
    count = Counter(result_to_show)
    all_num = len(result_to_show)
    sizes = [count[i] for i in range(8)]

    plt.figure(figsize=(15, 9))
    labels = ['Waiter', 'Cashier in a small supermarket', 'Sanitation workers', 'Takeout delivery staff', 'Tutoring',
              'Customer service', 'Guide', 'Caddy']

    colors = ['red', 'yellowgreen', 'lightskyblue', 'tomato', 'purple', 'brown', 'pink', 'olive']

    explode = (0, 0, 0, 0, 0, 0, 0, 0)
    patches, l_text, p_text = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      labeldistance=1.1, autopct='%3.1f%%', shadow=False,
                                      startangle=90, pctdistance=0.6)


    for t in l_text:
        t.set_size = (30)
    for t in p_text:
        t.set_size = (20)

    plt.axis('equal')
    plt.legend(loc='lower left')
    plt.title("Proportion of Predicted Jobs", fontsize='xx-large', fontweight='heavy', verticalalignment='bottom')
    plt.savefig("Proportion of Predicted Jobs.png")
    plt.show()

x_test_best = np.loadtxt('x_test_best.txt')
y_test_best = np.loadtxt('y_test_best.txt')

entropy_vector = np.loadtxt('entropy_vector.txt')
x = x_test_best[0]
x_reshape = np.reshape(x, (8, 7))
scores = [np.dot(x_reshape[i], entropy_vector) for i in range(len(x_reshape))]
scores = [round(scores[i], 2) for i in range(len(scores))]


print('hello world')
