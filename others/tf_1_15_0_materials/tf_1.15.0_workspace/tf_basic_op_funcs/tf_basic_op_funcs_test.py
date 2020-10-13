# coding=gbk
import tensorflow as tf
#***************************************1、tensorflow的基本运作 testing start ***************************************
#定义‘符号’变量，也称为占位符
a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b) #构造一个op节点

sess = tf.Session()#建立会话
#运行会话，输入数据，并计算节点，同时打印结果
print('a x b = ', sess.run(y, feed_dict={a: 3, b: 3}))
# 任务完成, 关闭会话.
sess.close()

#***************************************1、tensorflow的基本运作 testing end   ***************************************

#***************************************2.1 建立图(Building Graphs) testing start   ***************************************

#***************************************2.1 建立图(Building Graphs) testing end     ***************************************
#class tf.Graph
#tensorflow运行时需要设置默认的图
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g

##也可以使用tf.get_default_graph()获得默认图，也可在基础上加入节点或子图
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()

#tf.Graph.as_default
#以下两段代码功能相同
#1、使用Graph.as_default():
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

#2、构造和设置为默认
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g

#tf.Graph.control_dependencies(control_inputs)
# 错误代码
def my_func(pred, tensor):
  t = tf.matmul(tensor, tensor)
  with tf.control_dependencies([pred]):
    # 乘法操作(op)没有创建在该上下文，所以没有被加入依赖控制
    return t

# 正确代码
def my_func(pred, tensor):
  with tf.control_dependencies([pred]):
    # 乘法操作(op)创建在该上下文，所以被加入依赖控制中
    #执行完pred之后再执行matmul
    return tf.matmul(tensor, tensor)

# tf.Graph.name_scope(name)
# 一个图中包含有一个名称范围的堆栈，在使用name_scope(...)之后，将压(push)新名称进栈中，
#并在下文中使用该名称
with tf.Graph().as_default() as g:
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"

  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"

    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"

    # Create a nested scope called "inner_1".
    with g.name_scope("inner"):
      nested_inner_1_c = tf.constant(30.0, name="c")
      assert nested_inner_1_c.op.name == "nested/inner_1/c"

      # Treats `scope` as an absolute name scope, and
      # switches to the "nested/" scope.
      with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"

        with g.name_scope(""):
          e = tf.constant(50.0, name="e")
          print('e.op.name:',  e.op.name)
          assert e.op.name == "e"


#tf.Tensor.get_shape()
c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(c.get_shape())
#==> TensorShape([Dimension(2), Dimension(3)])

#现在有个用于图像处理的tensor->image
image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3], name='InputNode')
print('image shape:', image.get_shape(), ', name:', image.name, ', image op name:', image.op.name)
#==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])
# 假如我们知道数据集中图像尺寸为28 x 28，那么可以设置
image.set_shape([28, 28, 3])
print(image.get_shape())
#==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])