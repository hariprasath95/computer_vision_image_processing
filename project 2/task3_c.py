
# data 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
COLORS = ["red","green","blue"]
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def prob_density_function(b,mean,cov):
    return np.exp(-0.5*b.shape[0]*np.log(2*np.pi)) * (np.power(np.linalg.det(cov),-0.5)) * (np.exp(-0.5*np.dot(np.dot((b-mean).transpose(),np.linalg.inv(cov)),(b-mean))))


def compute_cluster_probablity(p,m,s):
    cluster_probablity = np.zeros((len(input_array),K))
    for i in range(len(input_array)):
        for j in range(K):
            cluster_probablity[i,j] = p[j]*prob_density_function(input_array[i],m[j],s[j])
        cluster_probablity[i] /= (np.sum(cluster_probablity[i]))
    return cluster_probablity
    
def compute_gmm(K,pi_value,mean,co_variance,num_iterns,showplot,points):
    for i in range(num_iterns):
        Q = compute_cluster_probablity(pi_value, mean, co_variance)
        new_pi_value = np.sum(Q,axis=0)/len(Q)
        numerator = zip(np.sum(Q*input_array[:,0][:,np.newaxis],axis=0),np.sum(Q*input_array[:,1][:,np.newaxis],axis=0))
        new_mu_value = numerator/(np.sum(Q,axis=0)[:,np.newaxis])
        computed_probablities = np.zeros([len(Q),K,2,2])
        for t in range(len(Q)):
            for j in range(K):
                dot_val1 = (input_array[t]-new_mu_value[j])[:,np.newaxis]
                dot_val2 = (input_array[t]-new_mu_value[j])[np.newaxis,:]
                computed_probablities[t,j] = Q[t,j] * np.dot(dot_val1,dot_val2)
        new_cov_value = np.sum(computed_probablities,axis=0)/np.sum(Q,axis=0)[:,np.newaxis,np.newaxis]
        pi_value = new_pi_value
        mean = new_mu_value
        co_variance = new_cov_value
        print("iteration",i+1)
        if(showplot):
            plt.clf()
            for j,x in enumerate(input_array):
                plt.scatter(x[0], x[1], edgecolor="red", facecolor='white', linewidth='1', marker=".")
            for j in range(3):
                plot_cov_ellipse(co_variance[j], mean[j], nstd=2,ax=None, alpha = 0.5,color = COLORS[j])
            plt.savefig("output/task3/task_3_gmm_iter"+str(i+1)+".png")
            plt.close()
    print("mean",mean)

    
# for task 1
input_array = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.7,3.4],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
K = 3
pi_value = np.array([1.0/3,1.0/3,1.0/3])
mean = np.array([[6.2,3.2],[6.6,3.7], [6.5,3.0]])
co_variance = np.array([[[0.5,0],[0,0.5]], [[0.5,0],[0,0.5]], [[0.5,0],[0,0.5]]])
compute_gmm(K,pi_value,mean,co_variance,1,False,None)

# for task 2
input_array =  np.array([[3.600, 79],[1.800, 54],[3.333, 74],[2.283, 62],[4.533, 85],[2.883, 55],[4.700, 88],[3.600, 85],[1.950, 51],[4.350, 85],[1.833, 54],[3.917, 84],[4.200, 78],[1.750, 47],[4.700, 83],[2.167, 52],[1.750, 62],[4.800, 84],[1.600, 52],[4.250, 79],[1.800, 51],[1.750, 47],[3.450, 78],[3.067, 69],[4.533, 74],[3.600, 83],[1.967, 55],[4.083, 76],[3.850, 78],[4.433, 79],[4.300, 73],[4.467, 77],[3.367, 66],[4.033, 80],[3.833, 74],[2.017, 52],[1.867, 48],[4.833, 80],[1.833, 59],[4.783, 90],[4.350, 80],[1.883, 58],[4.567, 84],[1.750, 58],[4.533, 73],[3.317, 83],[3.833, 64],[2.100, 53],[4.633, 82],[2.000, 59],[4.800, 75],[4.716, 90],[1.833, 54],[4.833, 80],[1.733, 54],[4.883, 83],[3.717, 71],[1.667, 64],[4.567, 77],[4.317, 81],[2.233, 59],[4.500, 84],[1.750, 48],[4.800, 82],[1.817, 60],[4.400, 92],[4.167, 78],[4.700, 78],[2.067, 65],[4.700, 73],[4.033, 82],[1.967, 56],[4.500, 79],[4.000, 71],[1.983, 62],[5.067, 76],[2.017, 60],[4.567, 78],[3.883, 76],[3.600, 83],[4.133, 75],[4.333, 82],[4.100, 70],[2.633, 65],[4.067, 73],[4.933, 88],[3.950, 76],[4.517, 80],[2.167, 48],[4.000, 86],[2.200, 60],[4.333, 90],[1.867, 50],[4.817, 78],[1.833, 63],[4.300, 72],[4.667, 84],[3.750, 75],[1.867, 51],[4.900, 82],[2.483, 62],[4.367, 88],[2.100, 49],[4.500, 83],[4.050, 81],[1.867, 47],[4.700, 84],[1.783, 52],[4.850, 86],[3.683, 81],[4.733, 75],[2.300, 59],[4.900, 89],[4.417, 79],[1.700, 59],[4.633, 81],[2.317, 50],[4.600, 85],[1.817, 59],[4.417, 87],[2.617, 53],[4.067, 69],[4.250, 77],[1.967, 56],[4.600, 88],[3.767, 81],[1.917, 45],[4.500, 82],[2.267, 55],[4.650, 90],[1.867, 45],[4.167, 83],[2.800, 56],[4.333, 89],[1.833, 46],[4.383, 82],[1.883, 51],[4.933, 86],[2.033, 53],[3.733, 79],[4.233, 81],[2.233, 60],[4.533, 82],[4.817, 77],[4.333, 76],[1.983, 59],[4.633, 80],[2.017, 49],[5.100, 96],[1.800, 53],[5.033, 77],[4.000, 77],[2.400, 65],[4.600, 81],[3.567, 71],[4.000, 70],[4.500, 81],[4.083, 93],[1.800, 53],[3.967, 89],[2.200, 45],[4.150, 86],[2.000, 58],[3.833, 78],[3.500, 66],[4.583, 76],[2.367, 63],[5.000, 88],[1.933, 52],[4.617, 93],[1.917, 49],[2.083, 57],[4.583, 77],[3.333, 68],[4.167, 81],[4.333, 81],[4.500, 73],[2.417, 50],[4.000, 85],[4.167, 74],[1.883, 55],[4.583, 77],[4.250, 83],[3.767, 83],[2.033, 51],[4.433, 78],[4.083, 84],[1.833, 46],[4.417, 83],[2.183, 55],[4.800, 81],[1.833, 57],[4.800, 76],[4.100, 84],[3.966, 77],[4.233, 81],[3.500, 87],[4.366, 77],[2.250, 51],[4.667, 78],[2.100, 60],[4.350, 82],[4.133, 91],[1.867, 53],[4.600, 78],[1.783, 46],[4.367, 77],[3.850, 84],[1.933, 49],[4.500, 83],[2.383, 71],[4.700, 80],[1.867, 49],[3.833, 75],[3.417, 64],[4.233, 76],[2.400, 53],[4.800, 94],[2.000, 55],[4.150, 76],[1.867, 50],[4.267, 82],[1.750, 54],[4.483, 75],[4.000, 78],[4.117, 79],[4.083, 78],[4.267, 78],[3.917, 70],[4.550, 79],[4.083, 70],[2.417, 54],[4.183, 86],[2.217, 50],[4.450, 90],[1.883, 54],[1.850, 54],[4.283, 77],[3.950, 79],[2.333, 64],[4.150, 75],[2.350, 47],[4.933, 86],[2.900, 63],[4.583, 85],[3.833, 82],[2.083, 57],[4.367, 82],[2.133, 67],[4.350, 74],[2.200, 54],[4.450, 83],[3.567, 73],[4.500, 73],[4.150, 88],[3.817, 80],[3.917, 71],[4.450, 83],[2.000, 56],[4.283, 79],[4.767, 78],[4.533, 84],[1.850, 58],[4.250, 83],[1.983, 43],[2.250, 60],[4.750, 75],[4.117, 81],[2.150, 46],[4.417, 90],[1.817, 46],[4.467, 74]])
K = 3
pi_value = np.array([1.0/3,1.0/3,1.0/3])
mean = np.array([[4.0,81],[2.0,57], [4.0,71]])
co_variance = np.array([[[1.30,13.98],[13.98,184.82]], [[1.30,13.98],[13.98,184.82]], [[1.30,13.98],[13.98,184.82]]])
compute_gmm(K,pi_value,mean,co_variance,5,True,input_array)
