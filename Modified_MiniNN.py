#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy 
import numpy as np
import numpy.random
numpy.set_printoptions(precision=3, floatmode="fixed")

class MiniNN: 
  """
  Naming convention: Any self variable starting with a capitalized letter and ending with s is a list of 1-D or 2-D numpy arrays, each element of which is about one layer, such as weights from one layer to its next layer. 
  self.Ws: list of 2-D numpy arrays, tranfer matrixes of all layers, ordered in feedforward sequence 
  self.phi: activation function 
  self.psi: derivative of activation function, in terms of its OUTPUT, ONLY when phi is logistic 
  self.Xs: list of 2-D numpy arrays, output from each layer
  self.Deltas: list of 2-D numpy arrays, delta from each layer
  """
  def logistic(self, x):
    return 1/(1 + numpy.exp(-x)) 

  def logistic_psi(self, x):
    """If the output of a logistic function is x, then the derivative of x over 
    the input is x * (1-x)
    """
    return x * (1-x)

  def __init__(self, input ,output, NeuronsInLayers): 
        
    """the user provides the number of non-bias neurons in each layer in form
        of a list. 
        list NeuronsInLayers elements represents the number of non-bais neurons in each layer.
        it keeps track of the size of the input and output to determine the dimensions of the transfer matrices after 
        the input layer and before the output layer.
       
        """
    """Initialize an NN
    hidden_layer: does not include bias 
    """
    Ws = [] #place holder
    L =len(NeuronsInLayers) #number of layer
    for n in range(L): 
      if n == 0:
        Ws.append(np.random.randn(len(input[0]),NeuronsInLayers[n])) # the transfer matrix for the first layer, input is augumented.
      else:
        Ws.append(np.random.randn(NeuronsInLayers[n - 1] + 1,NeuronsInLayers[n])) # +1 because of the augmnetion, output of previous layer is some value x n_neuornsL1 + 1.
     
     
    Ws.append(np.random.randn(NeuronsInLayers[n] + 1, len(output[0])))  #last layer transfer matrix.
    
    self. Ws = Ws
    self.phi = self.logistic # same activation function for all neurons
    self.psi = self.logistic_psi

  def feedforward(self, x, W, phi):
      """feedforward from previou layer output x to next layer via W and Phi
      return an augmented out where the first element is 1, the bias 
      Note the augmented 1 is redundant when the forwarded layer is output. 
      x: 1-D numpy array, augmented input
      W: 2-D numpy array, transfer matrix
      phi: a function name, activation function
      """

      return  numpy.concatenate(([1], # augment the bias 1
              phi(
                    numpy.matmul( W.transpose(), x )  
                ) # end of phi
              )) # end of concatenate

  def predict(self, X_0):
    """make prediction, and log the output of all neurons for backpropagation later 
    X_0: 1-D numpy array, the input vector, AUGMENTED
    """
    Xs = [X_0]; X=X_0
   
    for W in self.Ws:
     
      X = self.feedforward(X, W, self.phi)
      Xs.append(X)
    self.Xs = Xs
    self.oracle = X[1:] # it is safe because Python preserves variables used in for-loops

  def backpropagate(self, delta_next, W_now, psi, x_now):
    """make on step of backpropagation 
    delta_next: delta at the next layer, INCLUDING that on bias term  
                (next means layer index increase by 1; 
                 backpropagation is from next layer to current/now layer)
    W_now: transfer matrix from current layer to next layer (e.g., from layer l to layer l+1)
    psi: derivative of activation function in terms of the activation, not the input of activation function
    x_now: output of current layer 
    """
    delta_next = delta_next[1:] # drop the derivative of error on bias term 

    # first propagate error to the output of previou layer
    delta_now = numpy.matmul(W_now, delta_next) # transfer backward
    # then propagate thru the activation function at previous layer 
    delta_now *= self.psi(x_now) 
    # hadamard product This ONLY works when activation function is logistic
    return delta_now

  def get_deltas(self, target):
    """Produce deltas at every layer 
    target: 1-D numpy array, the target of a sample 
    delta : 1-D numpy array, delta at current layer
    """
    delta = self.oracle - target # delta at output layer is prediction minus target 
                                 # only when activation function is logistic 
    delta = numpy.concatenate(([0], delta)) # artificially prepend the delta on bias to match that in non-output layers. 
    self.Deltas = [delta] # log delta's at all layers

    for l in range(len(self.Ws)-1, -1, -1): # propagate error backwardly 
      # technically, no need to loop to l=0 the input layer. But we do it anyway
      # l is the layer index 
      W, X = self.Ws[l], self.Xs[l]  
      delta = self.backpropagate(delta, W, self.psi, X)
      self.Deltas.insert(0, delta) # prepend, because BACK-propagate
    
  def print_progress(self):
    """print Xs, Deltas, and gradients after a sample is feedforwarded and backpropagated 
    """
    print ("\n prediction: ", self.oracle)
    for l in range(len(self.Ws)+1): 
      print ("layer", l)
      print ("        X:", self.Xs[l], "^T")
      print ("    delta:", self.Deltas[l], "^T")
      if l < len(self.Ws): # last layer has not transfer matrix
        print ('        W:', numpy.array2string(self.Ws[l], prefix='        W: '))
      try: # because in first feedforward round, no gradient computed yet
           # also, last layer has no gradient
        print(' gradient:', numpy.array2string(self.Grads[l], prefix=' gradient: '))
      except: 
        pass
      
  def compute_grad(self):
    """ Given a sequence of Deltas and a sequence of Xs, compute the gradient of error on each transform matrix.
   Note that the first element on each delta is on the bias term. It should not be involved in computing the gradient on any weight because the bias term is not connected with previous layer. 
    """
    """ We modified the function 'update_weights' to the 'compute_grad' that only computes the gradient
    of error on each transform matrix.
    
    """
    self.Grads = []
    
    for l in range(len(self.Ws)): # l is layer index
      x = self.Xs[l]
      delta = self.Deltas[l+1]
      # print (l, x, delta)
      gradient = numpy.outer(x, delta[1:])
      self.Ws[l] -= 1 * gradient  # descent! 
      self.Grads.append(gradient)
      
    #print(self.Grads)  
    
    return self.Grads

  def update(self, grad):
        
    """ this function updates the weights given the gradients.
    this function is called at the end of the training of each batch to 
    enable batch update.
      
    """
        
        
    for l in range(len(self.Ws)):
        self.Ws[l] -= 1 * grad[l] # descent!
    
    
    # show that the new prediction will be better to help debug
    # self.predict(self.Xs[0])
    # print ("new prediction:", self.oracle)

  def train(self, X, Y, max_iter=100, verbose=False,batchSize = 1):
    """feedforward, backpropagation, and update weights
    The train function updates an NN using one sample. 
    Unlike scikit-learn or Tensorflow's fit(), x and y here are not a bunch of samples. 
    Homework: Turn this into a loop that we use a batch of samples to update the NN. 
    x: 2-D numpy array, an input vector
    y: 1-D numpy array, the target
    """
    """ The updated code: 
    feedforward, backpropagation, compute_grad and update.
    The train function updates an NN using batches of samples. 
    x and y here are bunch of samples. 
    
    x: 2-D numpy array, an input matrix
    y: 1-D numpy array, the target
    batch size: is defined by the user, default is 1.
    
    """
    
    
   
    for epoch in range(max_iter):   
        print ("epoch", epoch, end=":")
        #print(self.Ws)
        GradientL = [] # place holder for the gradient in each layer.
        
        for j in range(0,len(X),batchSize): # divide the input into batches
            x = X[j:j+batchSize]
            y = Y[j:j+batchSize]  
            for i in range(len(x)): # loops through the samples in a batch.
                self.predict(x[i]) # forward 
                print (self.oracle)
                self.get_deltas(y[i]) # backpropagate
                if verbose:
                    self.print_progress() 
                if (i==0):
                    GradientL = self.compute_grad()# compute gradients for the first sample in the batch. 
                    #print(GradientL)
                else:
                    GradSum = [] # place holder for the sum of greadients per layer.
                    for h,m in zip(GradientL,self.compute_grad()):
                        #print(h,m)
                        GradSum.append(np.add(h,m)) # sums the gradients per layer for each sample.
                        #print(GradSum)
                    GradientL = GradSum
            self.update(GradientL)  # updates the weights by the sum of gradients for each sample in the batch.  
        #print(self.Ws)
      
            
if __name__ == "__main__": 
  # The training sample
  x_0 = numpy.array(([[1, 1,3], [1,0,0], [1,4,5], [1,0,0]])) # input matrix, augmented 
  y_0 = numpy.array(([[0],[1],[0],[1]]))# output, target.
                          # this number must be between 0 and 1 because we used logistic activation and cross entropy loss. 
  # To use functions individually 
  #MNN = MiniNN(x_0,y_0,10,7)
  #Ws = MNN.Ws
  #MNN.predict(x_0)
  #MNN.get_deltas(y_0)
  #MNN.print_progress()
  #MNN.update_weights()
  #MNN.print_progress()

  # Or a recursive training process 
  MNN = MiniNN(x_0,y_0,[2,2]) # re-init

  MNN.train(x_0, y_0, max_iter=20,verbose=True,batchSize =2)
  


# In[ ]:




