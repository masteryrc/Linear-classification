import numpy
import matplotlib.pyplot

class Layer:
    # 初始化w、b矩阵，采取随机数初始化，这样可防止除0问题
    def __init__(self, dim_in, dim_out):
        self.weight = numpy.matrix(numpy.random.rand(dim_in, dim_out))
        self.bias = numpy.zeros(dim_out)
        self.delta = numpy.transpose(numpy.mat((numpy.ones(500*2))))

    # 计算输出
    def compute(self, x):
        sum = x * self.weight + self.bias
        return sum

    # 一次学习，学习率为rate
    def learn(self, x, y, rate):
        x = numpy.matrix(x, copy=False)
        y1 = self.compute(x)
        y = numpy.mat(y)
        # 依据差值delta修正w和b
        self.delta = y - y1 
        self.weight = self.weight + numpy.transpose(x) * self.delta * rate
        self.bias = self.bias + numpy.ones([1, x.shape[0]]) * self.delta * rate

if  __name__ == '__main__':
    a = Layer(2,1)
    num_observations = 3
    
    numpy.random.seed(12)
    num_observations = 500
    x1 = numpy.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = numpy.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = numpy.vstack((x2,x1)).astype(numpy.float32)
    www = numpy.hstack((numpy.ones(num_observations)))
    Y = numpy.transpose(numpy.mat(numpy.hstack((numpy.ones(num_observations),numpy.zeros(num_observations)))))
    """
    # 5000次学习
    for i in range(5000):
        #print(a.compute(X))
        因为数据未进行预处理，delta矩阵和x矩阵乘积是一个较大的数，因此取较小学习率，以保证收敛
        a.learn(X, Y, 0.0001)
        #print(a.weight,"\n", a.bias,"\n")
    """
    # k记录误差绝对值之和，当误差和足够小给出输出
    k = 1000
    while (k > 0.001):
        k = abs(a.delta.sum())
        #print(k)
        a.learn(X, Y, 0.0001)


    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(X[:num_observations,0], X[:num_observations,1], "bo")
    ax.plot(X[num_observations:,0], X[num_observations:,1], "ro")
    xs = numpy.arange(-4, 5)
    w=a.weight
    # 划线
    ys = (0.5 - a.bias[0,0] - xs * a.weight[0,0]) / a.weight[1,0]
    ax.plot(xs, ys, "k--")
    matplotlib.pyplot.show()

    print("ok!")
