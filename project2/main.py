import tensorflow as tf
from model import create_model_one, build_model_two
from util import load_preprocess_data, plot_history, plot_conf_matrix
import numpy as np


print("loading data")
(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_preprocess_data()


num_classes = 100
input_shape = (32, 32, 3)

class_names = [str(i) for i in range(num_classes)]
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#building models
print("Training model1 with adam\n")
model1 = create_model_one(input_shape=input_shape, num_classes=num_classes)
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
history1 = model1.fit(train_x, train_y, epochs=30, batch_size=64, 
                      validation_data=(val_x, val_y), callbacks=[early_stop])
plot_history(history1, 'Model1 Adam')

print("\nTraining model1 with sgd\n")
model1_sgd = create_model_one(input_shape=input_shape, num_classes=num_classes)
model1_sgd.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
history1_sgd = model1_sgd.fit(train_x, train_y, epochs=30, batch_size=64, 
                              validation_data=(val_x, val_y), callbacks=[early_stop])
plot_history(history1_sgd, 'Model1 SGD')


#second model adam

print("\nTraining model2 with adam \n")
model2 = build_model_two(input_shape=input_shape, num_classes=num_classes)
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
history2 = model2.fit(train_x, train_y, epochs=30, batch_size=64, 
                      validation_data=(val_x, val_y), callbacks=[early_stop])
plot_history(history2, 'Model2 Adam')

print("\nTraining model2 with sgd\n")
model2_sgd = build_model_two(input_shape=input_shape, num_classes=num_classes)
model2_sgd.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
history2_sgd = model2_sgd.fit(train_x, train_y, epochs=30, batch_size=64, 
                              validation_data=(val_x, val_y), callbacks=[early_stop])
plot_history(history2_sgd, 'Model2 SGD')

#testing best models
print("\nEvaluating best models")
test_loss, test_acc = model2.evaluate(test_x, test_y)
print(f"Test accuracy: {test_acc:.4f}")


print("\nMaking predictions / plotting confusion matrix")
y_pred = np.argmax(model2.predict(test_x), axis=1)
plot_conf_matrix(test_y, y_pred, class_names)
