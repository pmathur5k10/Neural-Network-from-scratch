#USE NUMPY FOR EXPONENTS,DOT OF MATRICES, ARRAY AND RANDOM NUMBERS
import numpy as np 

#ENTER THE INPUT AND OUTPUT SAMPLES FOR TRAINING
inar=np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outar=np.array([[0, 1, 1, 0]]).T

#CREATE A RANDOM NUMBER SEED TO GIVE THE SAME RANDOM NUMBER EVERY TIME
np.random.seed(1)

#CREATE TWO SYNAPSES TO BE APPLIED IN INITIAL AND INTERMEDIATE STAGES OF THE NEURAL NETWORK    	
synp1=2*np.random.random((3,4))-1
synp2=2*np.random.random((4,1))-1

#SIGMOID FXN
def sigmoid(x):

	return 1/(1+np.exp(-x));

#DERIVATIVE OF SIGMOID FXN
def sigmoidPrime(x):

	return x*(1-x)

#TRAINING OF THE DATA FOR 500000 ITERATIONS 
for i in range(50000):
#FORWARD PROPOGATION	
	l0=inar
	l1=sigmoid(np.dot(l0,synp1))
	l2=sigmoid(np.dot(l1,synp2))

#DISPLAY INITIAL STAGE OF NEURAL NETWORK TO COMPARE THE IMPROVEMENT	
	if(i==1):
		print (l2)


#BACK PROPOGATION		
	l2_error=outar-l2

	l2_delta=l2_error*sigmoidPrime(l2)

	l1_error=l2_delta.dot(synp2.T)

	l1_delta=l1_error*sigmoidPrime(l1)

	synp2+=l1.T.dot(l2_delta)
	synp1+=l0.T.dot(l1_delta)

#DISPLAY THE RESULT		
print (l2)




	
	

		

			
	









	











    
   




		



