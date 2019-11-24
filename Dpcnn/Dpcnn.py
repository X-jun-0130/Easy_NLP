import tensorflow as tf
import numpy as np
from data_processing import process, get_wordid, get_word2vec, read_category

###################################Parameters#########################
embedding_size = 100     #dimension of word embedding
vocab_size = 10000       #number of vocabulary

seq_length = 300         #max length of cnnsentence
num_classes = 10         #number of labels

num_filters = 250        #number of convolution kernel
kernel_size = 3          #size of convolution kernel

lr= 0.001                #learning rate

num_epochs = 5           #epochs
batch_size = 64          #batch_size

train_filename='./data/cnews.train.txt'  #train data
test_filename='./data/cnews.test.txt'    #test data
val_filename='./data/cnews.val.txt'      #validation data
vocab_filename='./data/vocab_word.txt'        #vocabulary
vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

categories, cat_to_id = read_category()
wordid = get_wordid(vocab_filename)
embeddings = get_word2vec(vector_word_npz)

############################################################

class data_loader():
    def __init__(self):

        self.train_x, self.train_y = process(train_filename, wordid, cat_to_id, max_length=300)
        self.test_x, self.test_y = process(test_filename, wordid, cat_to_id, max_length=300)
        self.train_x = self.train_x.astype(np.int32)
        self.test_x = self.test_x.astype(np.int32)
        self.train_y = self.train_y.astype(np.float32)
        self.test_y = self.test_y.astype(np.float32)
        self.num_train, self.num_test = self.train_x.shape[0], self.test_x.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        self.db_train = self.db_train.shuffle(self.num_train).batch(batch_size, drop_remainder=True)
        self.db_test = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))
        self.db_test = self.db_test.shuffle(self.num_test).batch(batch_size, drop_remainder=True)


    def get_batch(self, batch_s):
        indics = np.random.randint(0, self.num_train, batch_s)
        return self.train_x[indics], self.train_y[indics]

class DPCnn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(10000, 100,
                                                   embeddings_initializer=tf.constant_initializer(embeddings),
                                                   trainable=False)
        #self.embedding = tf.keras.layers.Embedding(10000, 100)
        self.regoin_embedding = tf.keras.layers.Conv2D(filters=num_filters,
                                                       kernel_size = [kernel_size,embedding_size],
                                                       strides=1,padding='valid',
                                                       activation=tf.nn.relu)
        self.con1v = tf.keras.layers.Conv2D(filters=num_filters,
                                            kernel_size = [3,1],
                                            strides=1,padding='same',
                                            activation=tf.nn.relu)
        self.max_pooling = tf.keras.layers.MaxPool2D(pool_size=(3,1), strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(36*1*250,))
        self.dense = tf.keras.layers.Dense(units=num_classes)


    def call(self, inputs):
        x_m = self.embedding(inputs)
        x_m = tf.expand_dims(x_m, -1) #expand dim
        x_re = self.regoin_embedding(x_m)
        x_1 = self.con1v(x_re)
        x_ = self.con1v(x_1)
        x = x_re + x_ #shortcut connection
        x = self.block(x)
        x = self.block(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        out = tf.nn.softmax(x)
        return out

    def block(self, inputs):
        x_m = self.max_pooling(inputs)
        x_1 = self.con1v(x_m)
        x_2 = self.con1v(x_1)
        x = x_m + x_2
        return x

def main():
    model = DPCnn()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(data_loader.db_train, epochs= num_epochs, validation_data=data_loader.db_test)
    model.evaluate(data_loader.db_test)


if __name__=='__main__':
    data_loader = data_loader()
    main()



# model = DPCnn()
# data_loader = data_loader()
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#
# for epoch in range(num_epochs):
#     print('Epoch:', epoch + 1)
#     num_batchs = int(data_loader.num_train / batch_size) + 1
#     for batch_index in range(num_batchs):
#         x, y = data_loader.get_batch(batch_size)
#         with tf.GradientTape() as tape:
#             y_pred = model(x)
#             correct_pre = tf.equal(tf.argmax(y_pred, 1), y)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#             loss = tf.reduce_mean(loss)
#             acuracy = tf.reduce_mean(tf.cast(correct_pre, 'float32'))
#             if batch_index % 100 == 0:
#                 print("batch %d: loss %f: accuracy %f" % (batch_index, loss.numpy(), acuracy.numpy()))
#         grads = tape.gradient(loss, model.variables)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
