import numpy as np

class GMM:
    '''
    Performs clustering of image data using a Gaussian Mixture Model.
    The model uses the EM algorithm to fit its parameters to the input image data.
    There is a multivariate Gaussian distribution for each cluster.
    The parameters fitted are the mean pixel value, the covariance matrix and the 
    weight of each Gaussian distribution.
    '''
    def __init__(self, n_clusters=3, max_iterations=100, tolerance=0.0015):
        '''
        Iniatializes the parameters of the EM algorithm.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters the algorithm is going to fit. The default is 3.
        max_iterations : int, optional
            The number of iterations the EM algorithm is going to execute. The default is 100.
        tolerance : float, optional
            The percentage of change between the centroids of conconsecutive iterations
            tolerated. If the change is smaller the algorithm will be considered as
            converged. The default is 0.0015.

        Returns
        -------
        None.

        '''
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def Initialize_parameters(self, data_points):
        '''
        Randomly initializes the parameters of each Gaussian distribution.\n
        It initializes n_cluster mean values of the same dimention as the datapoints.
        It initializes n_cluster covariance matrices. Each of these matrices is a 
        diagonal square matrix of the dimention of the datapoints.
        It initializes n_cluster weights for each distribution. The weights sum to one.
        
        Parameters
        ----------
        data_points : ndarray
            A 2D numpy array of all the datapoints and the dimention of the data.

        Returns
        -------
        None.

        '''
        from sklearn.datasets import make_spd_matrix
        self.means = np.random.permutation(data_points)[:self.n_clusters]
        self.covariances = np.zeros((self.n_clusters, data_points.shape[-1], data_points.shape[-1]))
        for c in range(self.n_clusters):
            self.covariances[c] = make_spd_matrix(data_points.shape[-1])
        self.weights = np.random.dirichlet(np.ones(self.n_clusters), size=1)[0]
        self.r = np.zeros((self.n_clusters, data_points.shape[0]))
        
    def N(self, x, mean, covariance):
        '''
        Calculates the probability value of the multivariate Gaussian distribution
        defined by the given mean and covariance matrix.

        Parameters
        ----------
        x : ndarray
            The data for which the propability is calculated.
        mean : ndarray
            The mean value of the distribution. The shape has to be the same as 
            that of the datapoints.
        covariance : ndarray
            A sqaure matrix of shape the same as that of the datapoints.
            The covariance matrix needs to be positively defined so that its 
            determinant is not zero.

        Returns
        -------
        ndarray
            The resulting probability values for x.

        '''
        d = len(x)
        return np.power(2 * np.pi, -d/2) * np.power(np.linalg.det(covariance), -1/2) * np.exp(-0.5 * (x - mean).T.dot(np.linalg.inv(covariance)).dot(x - mean))
        
    def E_Step(self, data_points):
        '''
        The Expectation step of the EM algorithm. This step calculates the responsibility
        of each datapoint for each cluster. What is the propability that each of
        the datapoints belongs to each cluster.

        Parameters
        ----------
        data_points : ndarray
            A 2D numpy array of all the datapoints and the dimention of the data.

        Returns
        -------
        None.

        '''        
        r = np.zeros((data_points.shape[0], self.n_clusters))
        for i in range(data_points.shape[0]):
            normalization = sum(self.weights[k] * self.N(data_points[i], self.means[k], self.covariances[k]) for k in range(self.n_clusters))
            for c in range(self.n_clusters):
                r[i][c] = self.weights[c] * self.N(data_points[i], self.means[c], self.covariances[c]) / normalization
        self.r = r
    
    def M_Step(self, r, data_points):
        '''
        This is the maximization step of the EM algorithm. This step updates the
        parameters of the Gaussian mixture model. It updates the means and the 
        covariances and the weights of each distribution based on the responibilities 
        that were calculated during the Expectation step.

        Parameters
        ----------
        r : ndarray
            The resopnsibilities calculated during the Expectation step.
        data_points : ndarray
            A 2D numpy array of all the datapoints and the dimention of the data..

        Returns
        -------
        None.

        '''
        m = sum(r)
        self.weights = m / self.n_clusters
        self.means = np.zeros(self.means.shape)
        self.covariances = np.zeros(self.covariances.shape)
        for i in range(data_points.shape[0]):
            for c in range(self.n_clusters):
                self.means[c] += r[i][c] * data_points[i] / m[c]
                temp = (data_points[i] - self.means[c]).reshape(1,-1)
                temp_t = (data_points[i] - self.means[c]).T.reshape(-1,1)
                self.covariances[c] += r[i][c] * temp_t.dot(temp) / m[c]
    
    def copy_means(self):
        '''
        Makes a copy of the means of the sistributions for comparison with the 
        changed means in the next iteration.

        Returns
        -------
        old_means : ndarray
            The means of the previous iteration.

        '''
        old_means = np.zeros(self.means.shape)
        for c in range(self.n_clusters):
            old_means[c] = self.means[c]
        return old_means
    
    def Convergence(self, old_means):
        '''
        Measures the change between the old and current means after each iteration.
        If the change is smaller than the tolerance then the EM algorithm has converged.

        Parameters
        ----------
        old_means : ndarray
            The means of the previous iteration.

        Returns
        -------
        convergence : Boolean
            A Boolean value of the convergence of the EM algorithm.

        '''
        convergence = False
        change = np.sum(np.abs(self.means - old_means) / (old_means * 100.0))
        if change < self.tolerance:
            convergence = True
        return convergence
    
    def fit(self, data):
        '''
        The distributions are fitted with the input image data using the EM algorithm.\n
        The parameters of the distributions are randomly initialized.\n
        The EM algorithm iterates a total of max_iterations, unless the convergence
        criterium is met.\n
        In each iteration the Expectation step and the Maximization step are executed.\n
        The means of each iteration are compared with the means of the previous 
        iteration, to check for convergence.

        Parameters
        ----------
        data : ndarray
            Input image data. This data is normalized before it is used in the
            EM algorithm.

        Returns
        -------
        None.

        '''
        data_points = data.reshape(-1, data.shape[-1]) / 255.0
        self.Initialize_parameters(data_points)
        
        for iteration in range(self.max_iterations):
            self.E_Step(data_points)
            old_means = self.copy_means()
            self.M_Step(self.r, data_points)
            if self.Convergence(old_means): break
            print('Iteration: ', iteration+1)
    
    def predict(self, data):
        '''
        Calculates which of the distributions best describes the input data.

        Parameters
        ----------
        data : ndarray
            Input image data.

        Returns
        -------
        ndarray
            A numpy array of the clustered image. Each pixel of the input image 
            data gets replaced by the mean of the distribution with the highest
            propability. This is the distribution that each pixel is most likely
            to be belong in.

        '''
        data_points = data.reshape(-1, data.shape[-1])
        clustered_data = np.zeros(data_points.shape)
        for i in range(data_points.shape[0]):
            ass = list(self.r[i])
            cluster = ass.index(max(ass))
            clustered_data[i] = self.means[cluster] * 255
        return clustered_data.reshape(data.shape).astype(int)