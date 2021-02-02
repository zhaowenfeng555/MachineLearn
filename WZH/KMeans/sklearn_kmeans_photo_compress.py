from skimage import io
from cv2 import *
from sklearn.cluster import KMeans
import numpy as np

class Compress(object):
    """
    compress
    """
    def __init__(self):
        """
        init
        """
        self.r = 0
        self.g = 0


        # 输入：  r * g * b
        # reshape : r*g, b
        # clusters_center:   k, b
        # labels: r*g
        # labels: r, g


    def compress(self):
        image = imread('./1.jpg')
        print(image.shape)
        print(image.dtype)
        print(image.size)

        r = image.shape[0]
        g = image.shape[1]
        b = image.shape[2]

        self.r = r
        self.g = g

        image = image.reshape(r * g, b)
        print (image)

        # (self, n_clusters=8, *, init='k-means++', n_init=10,
        # max_iter=300, tol=1e-4, precompute_distances='deprecated',
        # verbose=0, random_state=None, copy_x=True,
        # n_jobs='deprecated', algorithm='auto'):

        kmeans = KMeans(n_clusters= 4, n_init=10, max_iter=3)
        kmeans.fit(image)

        clusters_center = np.asanyarray(kmeans.cluster_centers_, dtype = np.uint8)
        labels = np.asarray(kmeans.labels_, dtype = np.uint8)
        print ('labels', ' ', labels)
        # for item in labels:
        #     print (item)
        print (labels.shape)
        labels = labels.reshape(r, g)
        print ('labels2', ' ', labels)
        print (labels.shape)

        print ('*****')
        print (clusters_center)

        np.save('compressed_cluster_center.npy', clusters_center)
        imwrite('comresssed_image_labels.png', labels)

    def reconstruct_image(self):
        """
        recon
        """
        clusters_centers = np.load('compressed_cluster_center.npy')
        labels = imread('comresssed_image_labels.png')
        print (labels)

        # image = np.zeros([labels.shape[0], labels[1], 3], dtype = np.uint8)
        image = np.zeros([self.r, self.g, 3], dtype = np.uint8)
        print ('shape', image.shape)
        print('labels', labels.shape)
        print (clusters_centers.shape)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                image[i, j, :] = clusters_centers[labels[i, j, 0], :]

        imwrite('1_reconstruce.jpg', image)

    def main(self):
        """
        main
        """
        self.compress()
        self.reconstruct_image()

if __name__ == '__main__':
    """
    main
    """
    com = Compress()
    com.main()
