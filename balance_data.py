import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
train_data = np.load('training_data.npy')
print((train_data))


df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
right = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img, choice])
    elif:
        choice == [0,0,1]:
        rights.append([img, choice])
    elif:
        choice == [0,1,0]:
        forwards.append([img, choice])
    else:
        print('task failed succesfully')

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(rights)]

final_data = forwards + lefts + rights
print(len(final_data))
np.save('training_data_v2.npy', final_data)


#for data in train_data:
#    img = data[0]
#    choice = data[1]
#    cv2.imshow('test', img)
#    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
