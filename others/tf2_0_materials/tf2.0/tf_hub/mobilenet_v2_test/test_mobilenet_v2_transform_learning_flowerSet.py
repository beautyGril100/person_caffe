# coding=gbk
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import time
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
 
def convert_h5to_pb(keras_h5_model):
  model = tf.keras.models.load_model(keras_h5_model,custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
  model.summary()
  full_model = tf.function(lambda Input: model(Input))
  full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  
  # Get frozen ConcreteFunction
  frozen_func = convert_variables_to_constants_v2(full_model)
  frozen_func.graph.as_graph_def()
  
  layers = [op.name for op in frozen_func.graph.get_operations()]
  print("-" * 50)
  print("Frozen model layers: ")
  for layer in layers:
      print(layer)
  
  print("-" * 50)
  print("Frozen model inputs: ")
  print(frozen_func.inputs)
  print("Frozen model outputs: ")
  print(frozen_func.outputs)
  
  # Save frozen graph from frozen ConcreteFunction to hard drive
  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./",
                    name=keras_h5_model.split('.', 1)[0] + '.pb',
                    as_text=False)


if __name__ == '__main__':
  #×××××××××Step 1 - tensorflow hub mobilenet_v2 test - *****************
  # 加载一个图像分类器
  classifier_url = 'https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/4'
  IMAGE_SHAPE = (224, 224)
  classifier = tf.keras.Sequential([
      hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
  ])
  
  # 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
  # grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
  grace_hopper = './test_images/dog.jpg'
  grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
  grace_hopper = np.array(grace_hopper)/255.0
  result = classifier.predict(grace_hopper[np.newaxis, ...])
  predicted_class = np.argmax(result[0], axis=-1)
  
  # 加标注
  # 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
  # labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
  labels_path = 'labels_1001_mobilenet_quant_v1_224.txt'
  imagenet_labels = np.array(open(labels_path).read().splitlines())
  # 可视化
  plt.imshow(grace_hopper)
  plt.axis('off')
  predicted_class_name = imagenet_labels[predicted_class]
  _ = plt.title("Prediction: " + predicted_class_name.title())
  plt.show()
  
  t = time.time()
  export_path_savedModel = "./keras_mobilenet_v2_classification_imageNet_saved_models/{}".format(int(t))
  classifier.save(export_path_savedModel, save_format='tf')
  export_path_h5 = 'keras_mobilenet_v2_classification_imageNet.h5'
  classifier.save(export_path_h5)
  _ = tf.keras.models.load_model(export_path_h5, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
  
  #×××××××××Step 2 - tensorflow hub mobilenet_v2 used to predict flower's class - *****************
  # 下载花数据集
  data_root = tf.keras.utils.get_file(
    'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
     untar=True)
  # 使用ImageDataGenerator's rescale 把数据转成tf hub需要的格式（值范围都在[0,1]之间）
  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
  image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
  
  image_batch, label_batch = image_data[0]
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  '''
  for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
  '''
  # 上面输出
  # Image batch shape:  (32, 224, 224, 3)
  # Label batch shape:  (32, 5)
  # 试试预测
  result_batch = classifier.predict(image_batch)
  predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
  plt.figure(figsize=(10,9))
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
  _ = plt.suptitle("ImageNet predictions")
  
  plt.show()
  
  #×××××××××Step 3 - tensorflow hub mobilenet_v2 used to predict flower's class with transform learning - *****************
  feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
  feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                           input_shape=(224,224,3))
  feature_batch = feature_extractor_layer(image_batch)
  print('feature_batch.shape:', feature_batch.shape)  # (32, 1280)
  feature_extractor_layer.trainable = False
  # 重新创建模型
  print('image_data.num_classes:', image_data.num_classes)
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
  ])
  # 预测器
  predictions = model(image_batch)
  # 编译模型
  model.compile(
    optimizer=tf.keras.optimizers.Adam(),                            
    loss='categorical_crossentropy',
    metrics=['acc'])
  
  # 训练
  class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
      self.batch_losses = []
      self.batch_acc = []
  
    def on_train_batch_end(self, batch, logs=None):
      self.batch_losses.append(logs['loss'])
      self.batch_acc.append(logs['acc'])
      self.model.reset_metrics()
  
  steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
  
  batch_stats_callback = CollectBatchStats()
  
  history = model.fit_generator(image_data, epochs=3,
                                steps_per_epoch=steps_per_epoch,
                                callbacks = [batch_stats_callback])
  # 显示训练时损失值变化
  plt.figure()
  plt.ylabel("Loss")
  plt.xlabel("Training Steps")
  plt.ylim([0,2])
  plt.plot(batch_stats_callback.batch_losses)
  plt.show()
  # 准确率变化情况
  plt.figure()
  plt.ylabel("Accuracy")
  plt.xlabel("Training Steps")
  plt.ylim([0,1])
  plt.plot(batch_stats_callback.batch_acc)
  plt.show()
  
  # 测试结果
  class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
  class_names = np.array([key.title() for key, value in class_names])
  predicted_batch = model.predict(image_batch)
  predicted_id = np.argmax(predicted_batch, axis=-1)
  predicted_label_batch = class_names[predicted_id]
  label_id = np.argmax(label_batch, axis=-1)
  # 可视化
  plt.figure(figsize=(10,9))
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
  _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
  
  plt.show()
  
  # save model
  t = time.time()
  export_path_savedModel = "./keras_mobilenet_v2_classification_flower_saved_models/{}".format(int(t))
  model.save(export_path_savedModel, save_format='tf')
  export_path_h5 = 'keras_mobilenet_v2_classification_flower.h5'
  model.save(export_path_h5)
  # 装载
  reloaded = tf.keras.models.load_model(export_path_savedModel)
  reloaded_h5 = tf.keras.models.load_model(export_path_h5, custom_objects={'KerasLayer': hub.KerasLayer})
  result_batch = model.predict(image_batch)
  reloaded_result_batch = reloaded.predict(image_batch)
  reloaded_h5_result_batch = reloaded_h5.predict(image_batch)
  print('reloaded_result_batch:', len(reloaded_result_batch))
  print('result_batch:', len(result_batch))
  diff_max = abs(reloaded_result_batch - result_batch).max()
  diff_max_h5 = abs(reloaded_h5_result_batch - reloaded_result_batch).max()
  if not diff_max and not diff_max_h5:
      print('Successful!!!')
  else:
      print('Error!!! diff_max is not equal to 0.')
      
  convert_h5to_pb(export_path_h5)






