from itechnique import ITechnique
import cv2
import numpy as np
import numpy as np

class zcaWhiteningAugmentationTechnique(ITechnique):

    # Valid values for pover are in the range (0.25,4]
    def __init__(self):
        ITechnique.__init__(self,False)


    def __zca_whitening_matrix(self,X):
        """
        Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
        INPUT:  X: [M x N] matrix.
            Rows: Variables
            Columns: Observations
        OUTPUT: ZCAMatrix: [M x M] matrix
        """
        # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
        sigma = np.cov(X, rowvar=True)  # [M x M]
        # Singular Value Decomposition. X = U * np.diag(S) * V
        U, S, V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
        # Whitening constant: prevents division by zero
        epsilon = 1e-5
        # ZCA Whitening matrix: U * Lambda * U'
        ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
        return ZCAMatrix

    def apply(self, image):
        ZCAMatrix = self.__zca_whitening_matrix(image)
        xZCAMatrix = np.dot(ZCAMatrix, image)
        return xZCAMatrix

technique = zcaWhiteningAugmentationTechnique()
image = cv2.imread("LPR1.jpg")
cv2.imshow("new",technique.applyForClassification(image))
cv2.waitKey(0)
