###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################
import numpy as np

def add_bias_term(X):
    """
    Add a bias term to each sample of the input data.
    """

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

class LogisticRegressionGD():
    """
    Logistic Regression Classifier.

    Fields:
    -------
    w_ : array-like, shape = [n_features]
      Weights vector, where n_features is the number of features.
    learning_rate : float
      Learning rate (between 0.0 and 1.0)
    max_iter : int
      Maximum number of iterations for gradient descent
    eps : float
      Minimum BCE loss reduction from previous iteration to declare convergence.
    random_state : int
      Random number generator seed for random weight
      initialization.
    loss_history_ : list of float or None
      BCE loss after each gradient-descent iteration; set by ``fit``.
    """
    
    def __init__(self, learning_rate=0.0001, max_iter=10000, eps=0.000001, random_state=1):
       
        # Initialize the weights vector with small random values
        self.random_state = random_state
        self.w_ = np.nan
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.class_names = None
        self.loss_history_ = None


    def predict_proba(self, X):
        """
        Return the predicted probabilities for the positive class (label 1) to all samples in X

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.

        Returns
        -------
        y_pred_prob : array-like, shape = [n_samples]
          Predicted probabilities (for class 1) for all the instances
        """
        class_1_prob = np.nan * np.ones(X.shape[0])
        assert X.shape[1] == self.w_.shape[0]

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return class_1_prob
        

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class label according to the threshold

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        threshold : float, optional
          Threshold for the predicted class label.
        
        Returns
        -------
        y_pred : array-like, shape = [n_samples]
          Predicted class labels for all the instances; must use the **same type and value** as in ``y_true`` passed to ``fit``.
        """
        y_pred = np.nan * np.ones(X.shape[0])
    
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return y_pred

    
    def BCE_loss(self, X, y):
        """
        Calculate the BCE loss (not needed for training)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels. 

        Returns
        -------
        BCE_loss : float
          The BCE loss.
          Make sure to normalize the BCE loss by the number of samples.
        """

        y_01 = np.where(y == self.class_names[0], 0, 1) # represents the class 0/1 labels
        loss = None
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return loss


    def fit(self, X, y):
        """ 
        Fit training data by minimizing the BCE loss using gradient descent.
        Updates the weight vector (field of the object) in each iteration using gradient descent.
        The gradient should correspond to the BCE loss.
        Stop the function when the difference between the previous BCE loss and the current is less than eps
        or when you reach max_iter.
        Record the BCE loss after each iteration in self.loss_history_ (similar to scikit-learn estimators).
        Note that this is the function that maps the class labels in y to {0,1}.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels.

        """

        # Initial weights are set in constructor
        self.loss_history_ = []

        # map classes to 0,1 (arbitrarily)
        self.class_names = np.unique(y)
        y_01 = np.where(y == self.class_names[0], 0, 1)
        np.random.seed(self.random_state)
        self.w_ = 1e-6 * np.random.randn(X.shape[1])

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        # Append each iteration's BCE loss to self.loss_history_.                 #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        


class LogisticRegressionGD_MB(LogisticRegressionGD):
    """
    Logistic Regression Classifier with mini-batch gradient descent.
    """
    def __init__(self, learning_rate=0.0001, eps=0.000001, random_state=1, batch_size=32, num_epochs=100):
        INF = np.inf
        super().__init__(learning_rate, max_iter=INF, eps=eps, random_state=random_state)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y, num_epochs=None, batch_size=None):
        """ 
        Fit training data by minimizing the BCE loss using mini-batch gradient descent.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels.
        num_epochs : int, optional
          The number of epochs to train the model. If None, use ``self.num_epochs``.
        batch_size : int, optional
          The number of samples in each mini-batch. If None, use ``self.batch_size``.
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
        if batch_size is None:
            batch_size = self.batch_size

        # make sure to use 0/1 labels:
        self.class_names = np.unique(y)
        y_01 = np.where(y == self.class_names[0], 0, 1)
        np.random.seed(self.random_state)
        self.w_ = 1e-6 * np.random.randn(X.shape[1])

        self.loss_history_ = []

        def shuffle_data(X, y, y_01):
          # usage: X_shuffled, y_shuffled, y_01_shuffled = shuffle_data(X, y, y_01)
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            y_01_shuffled = y_01[permutation]
            return X_shuffled, y_shuffled, y_01_shuffled

        

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        # Append each iteration's BCE loss to self.loss_history_.                 #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
      
def calc_metrics(y_true, y_pred, positive_class):
    """
    Calculate the metrics for the LogisticRegression classifier.
    Parameters
    ----------
    y_true : {array-like}, shape = [n_samples]
           True class labels for samples in dataset
    y_pred : {array-like}, shape = [n_samples]
           Predicted class labels for samples in dataset
    positive_class : the class label to consider as positve 
           (the other label will be considered as negative)
           Must use the **same type and value** as in ``y_true``
    Returns
    -------
    metric_dict : a dictionary with metric names and values
    """

    metric_dict = dict(tp=None, fp=None, tn=None, fn=None, tpr=None, fpr=None,
                       tnr=None, fnr=None, accuracy=None, precision=None, risk=None, f1=None)
    # Calculate the metrics
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
   
    return metric_dict


def fpr_tpr_per_threshold(y_true, positive_class_probs, positive_class):
    """
    Calculate FPR and TPR of a given classifier for different thresholds

    Parameters
    ----------
    y_true : {array-like}, shape = [n_samples]
           True class labels for samples in dataset
    positive_class_probs : {array-like}, shape = [n_samples]
           Predicted probabilities for the positive class for samples in dataset
    positive_class : the class label to consider as positve 
           (the other label will be considered as negative)
           Must use the **same type and value** as in ``y_true``

    We say that sample i is predicted as positive under threshold t if:
      positive_class_probs[i] >= t 
    """
    fpr = []
    tpr = []
    # consider thresholds from 0 to 1 with step 0.01
    prob_thresholds = np.arange(0, 1, 0.01)
    y_true = np.asarray(y_true)
    if not np.any(y_true == positive_class):
        raise ValueError(
            "No entry in y_true equals positive_class under ==. "
            "Match types: use e.g. positive_class=8 for integer labels or positive_class='8' for strings."
        )
    if np.all(y_true == positive_class):
        raise ValueError(
            "All labels equal positive_class; ROC needs both positive and negative samples."
        )
    y_true_binary = np.where(y_true == positive_class, 1, 0)
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fpr, tpr


class OneVsAllClassifier:
    """
    One-vs-all multi-class classifier.

    This classifier receives a binary classifier class and trains one binary
    classifier per class. The classifier for class c is trained to distinguish
    class c from all other classes.

    Fields:
    -------
    binary_classifier_class : class
      The binary classifier class to instantiate for each one-vs-all problem.
      The class must implement fit and predict_proba.
    binary_classifier_kwargs : dict
      Keyword arguments passed to each binary classifier constructor.
    classes_ : array-like, shape = [n_classes]
      Sorted class labels seen during fit.
    classifiers_ : list
      The fitted binary classifiers, one per class in classes_.
    """

    def __init__(self, binary_classifier_class=LogisticRegressionGD_MB, **binary_classifier_kwargs):
        self.binary_classifier_class = binary_classifier_class
        self.binary_classifier_kwargs = binary_classifier_kwargs
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X, y):
        """
        Fit one binary classifier per class.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors.
        y : array-like, shape = [n_samples]
          Multi-class labels.
        """
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        # For each class, create binary labels with 1 for that class and 0 for    #
        # all other classes. Fit a new binary classifier and append it to         #
        # self.classifiers_.                                                     #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return self

    def predict_proba(self, X):
        """
        Return one-vs-all positive-class scores for every class.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors.

        Returns
        -------
        scores : array-like, shape = [n_samples, n_classes]
          scores[i, j] is the positive-class probability predicted by the
          classifier trained for classes_[j].
        """
        scores = np.nan * np.ones((X.shape[0], len(self.classes_)))

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        # Fill each column with the positive-class probabilities from the          #
        # corresponding binary classifier.                                        #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return scores

    def predict(self, X):
        """
        Return the class with the largest one-vs-all score for each sample.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors.
        """
        y_pred = np.nan * np.ones(X.shape[0])

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return y_pred
