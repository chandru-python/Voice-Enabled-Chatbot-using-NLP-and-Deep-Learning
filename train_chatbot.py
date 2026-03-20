import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('new.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lemmatized words")

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

X_train = np.array([i[0] for i in training])
y_train = np.array([i[1] for i in training])

print("Training data created")

# MODEL (UNCHANGED)
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    "best_chatbot_model.h5",
    monitor="accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

hist = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=32,
    verbose=1,
    callbacks=[checkpoint]
)

# -----------------------------
# FINAL ACCURACY
# -----------------------------
loss, accuracy = model.evaluate(X_train, y_train)
print("\nFinal Training Accuracy:", accuracy)

# -----------------------------
# PRECISION / RECALL / F1
# -----------------------------
y_pred = model.predict(X_train)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_train, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# -----------------------------
# GRAPH
# -----------------------------
plt.figure()
plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

print("Best model saved as best_chatbot_model.h5")
