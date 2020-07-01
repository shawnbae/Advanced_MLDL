import tensorflow as tf

# Tensor constant
t1= tf.constant([[[1,1,1],[2,2,2]],
                 [[3,3,3],[4,4,4]]],dtype= 'float64')
t2= tf.constant([[[4,4,4],[3,3,3]],
                  [[2,2,2],[1,1,1]]],dtype= 'float64')

# Tensor Variable
w= tf.Variable([[1,1],[2,2]],dtype= 'float64')

# Tensor dtype change
tf.dtypes.cast(t1, 'int32')
tf.dtypes.cast(t2, 'int32')

# Tensor element-wise add/square/sqrt
tf.add(t1,t2)
tf.square(t1)
tf.sqrt(t1)

# generate zero elements tensor
tf.zeros(shape= (3,2,2))

# generate one elements tensor
tf.ones(shape= (3,2,2))

# generate one-hot tensor
tf.one_hot(indices= [0,1,2], depth= 3)

# Tensor shape
t1.shape # 행,렬,elements
tf.size(t1) # 총 size

# matrix multiply
tf.matmul(w,t1)

# 축별 합
tf.reduce_sum(t1,axis=0)
tf.reduce_sum(t1,axis=1)
tf.reduce_sum(t1,axis=2)

# 축별 평균
tf.reduce_mean(t1,axis=0)

# sorting tensor
tf.sort()

# concatenate
tf.concat(tf.dtypes.cast(t1,'int32'),tf.dtypes.cast(t2,'int32'))

# tf.where -> condition을 통해 index를 출력할 수 있음.
tf.where(t1==2)

# slicing tensor by shape
tf.slice(t1,[1,0,0],[1,2,2])

# clip the value by min and max
# 주로 log0이 되는 것을 방지할 때 사용
tf.clip_by_value(t1,1.5,3.5)