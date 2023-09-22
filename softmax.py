#!/usr/bin/env python
# coding: utf-8

# ## softmax regression

# In[6]:


import tensorflow as tf
from tensorflow import keras


# In[7]:



from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# ## preprocess data

# In[8]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# ## softmax regression

# In[9]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  
    keras.layers.Dense(10, activation='softmax') 
])


# In[10]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[11]:


model.fit(train_images, train_labels, epochs=5)


# In[12]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[13]:


print(f"Test accuracy: {test_acc * 100:.2f}%")


# In[ ]:




