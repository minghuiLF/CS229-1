from matplotlib.image import  imread
import matplotlib.pyplot as plt
import numpy as np

def dis(a,b):
    res = 0
    for i in range(a.shape[0]):
        res += (a[i]-b[i])**2
    return res

if __name__ == '__main__':
    # read the image
    A = imread('mandrill-large.tiff')
    B = np.zeros(A.shape,dtype=A.dtype)
    plt.imshow(A)
    plt.show()

    # k-means algorithm
    cluster_num = 16

    centroids = np.zeros([cluster_num,3])
    for k in range(cluster_num):
        centroids[k] = A[np.random.randint(0,A.shape[0])][np.random.randint(0,A.shape[1])]
    iter = 0
    tot_iter = 30
    distortion_history = []
    last_distortion = -1
    while iter < tot_iter:
        iter+=1
        close_points = []
        for k in range(cluster_num):
            close_points.append([])
        distortion = 0

        # find the closest centroid
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                min_dis = dis(A[i][j],centroids[0])
                min_num = 0
                for k in range(cluster_num):
                    cur_dis = dis(A[i][j],centroids[k])
                    if cur_dis < min_dis:
                        min_num = k
                        min_dis = cur_dis
                close_points[min_num].append((i,j))
                distortion += min_dis

        # update centroids
        for k in range(cluster_num):
            if len(close_points[k])!=0:
                centroids[k] = np.zeros(centroids[k].shape[0])
                for (i,j) in close_points[k]:
                    centroids[k] += A[i][j]
                centroids[k]/=len(close_points[k])

        # print iteration information
        print('Iteration %d Distortion %.4f'%(iter,distortion))
        if last_distortion>=0:
            distortion_history.append(last_distortion-distortion)
        last_distortion = distortion

        # replace with centroids
        if iter == tot_iter:
            for k in range(cluster_num):
                if len(close_points[k]) != 0:
                    for (i, j) in close_points[k]:
                        B[i][j] = centroids[k]
            plt.title('Replaced Image')
            plt.imshow(B)
            plt.show()


    plt.title('Distortion Change')
    plt.plot(np.arange(1,iter,1),distortion_history)
    plt.show()


