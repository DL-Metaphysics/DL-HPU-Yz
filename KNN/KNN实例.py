from numpy import *
import operator

dataSet = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']

def classify0(inX,dataSet,labels,k):

    #求出样本集的行数，也就是labels标签的数目
    dataSetSize = dataSet.shape[0]

    #构造输入值和样本集的差值矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet

    #计算欧式距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    #求距离从小到大排序的序号
    sortedDistIndicies = distances.argsort()

    #对距离最小的k个点统计对应的样本标签
    classCount = {}
    for i in range(k):
        #取第i+1邻近的样本对应的类别标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #以标签为key，标签出现的次数为value将统计到的标签及出现次数写进字典
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    #对字典按value从大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    #返回排序后字典中最大value对应的key
    return sortedClassCount[0][0]
if __name__ == '__main__':
    print(classify0([1.1,0],dataSet,labels,3))
