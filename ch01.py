import scipy as sp 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#Aim to answer the question: how long will our server handle the incoming web traffic?
#To answer this we have to: 1. find the real model behind noisy data
#2.Use model to extrapolate into the future to find point where infrastructure has to be extended
def main():
	data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
	data.shape(743, 2)

	#x will contain the hours column 0 from the data set
	x = data[:, 0]
	#y will contain the web hits column 1 in that particular hour
	y = data[:, 1]

	#logically negate so that we choose only those elements from x and y
	#where y does contain valid numbers
	x = x[~sp.isnan(y)]
	y = y[~sp.isnan(y)]

	#plotting data
	plt.scatter(x,y)
	plt.title("Web traffic over the last month")
	plt.xlabel("Time")
	plt.ylabel("Hits/Hour")
	plt.xticks([w*7*24 for w in range(10)],
	 ['week %i'%w for w in range(10)])
	plt.autoscale(tight=True)
	plt.grid()

	#after trying different models we avoid overfitting and underfitting
	f2p = sp.polyfit(x, y, 2)
	f2 = sp.poly1d(f2p)
	reached_max = fsolve(f2-100000, 800)/(7*24)



#returns the error as squared distance of the model's prediction to the real data
#f is the learned model function
def error(f, x, y):
	return sp.sum((f(x) - y)**2)

if __name__ == "__main__":
    main()