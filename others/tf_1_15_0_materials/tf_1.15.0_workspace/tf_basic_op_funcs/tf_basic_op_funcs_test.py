# coding=gbk
import tensorflow as tf
#***************************************1��tensorflow�Ļ������� testing start ***************************************
#���塮���š�������Ҳ��Ϊռλ��
a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b) #����һ��op�ڵ�

sess = tf.Session()#�����Ự
#���лỰ���������ݣ�������ڵ㣬ͬʱ��ӡ���
print('a x b = ', sess.run(y, feed_dict={a: 3, b: 3}))
# �������, �رջỰ.
sess.close()

#***************************************1��tensorflow�Ļ������� testing end   ***************************************

#***************************************2.1 ����ͼ(Building Graphs) testing start   ***************************************

#***************************************2.1 ����ͼ(Building Graphs) testing end     ***************************************
#class tf.Graph
#tensorflow����ʱ��Ҫ����Ĭ�ϵ�ͼ
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g

##Ҳ����ʹ��tf.get_default_graph()���Ĭ��ͼ��Ҳ���ڻ����ϼ���ڵ����ͼ
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()

#tf.Graph.as_default
#�������δ��빦����ͬ
#1��ʹ��Graph.as_default():
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

#2�����������ΪĬ��
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g

#tf.Graph.control_dependencies(control_inputs)
# �������
def my_func(pred, tensor):
  t = tf.matmul(tensor, tensor)
  with tf.control_dependencies([pred]):
    # �˷�����(op)û�д����ڸ������ģ�����û�б�������������
    return t

# ��ȷ����
def my_func(pred, tensor):
  with tf.control_dependencies([pred]):
    # �˷�����(op)�����ڸ������ģ����Ա���������������
    #ִ����pred֮����ִ��matmul
    return tf.matmul(tensor, tensor)

# tf.Graph.name_scope(name)
# һ��ͼ�а�����һ�����Ʒ�Χ�Ķ�ջ����ʹ��name_scope(...)֮�󣬽�ѹ(push)�����ƽ�ջ�У�
#����������ʹ�ø�����
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

#�����и�����ͼ�����tensor->image
image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3], name='InputNode')
print('image shape:', image.get_shape(), ', name:', image.name, ', image op name:', image.op.name)
#==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])
# ��������֪�����ݼ���ͼ��ߴ�Ϊ28 x 28����ô��������
image.set_shape([28, 28, 3])
print(image.get_shape())
#==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])