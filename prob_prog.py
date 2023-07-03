from copy import deepcopy
import numpyro
import jax
import numpyro.infer
import numpyro.optim
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp
# import numpy as np
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO
from multiprocessing import Pool
import time
import os
import numpy as np
import re
import copy
import subprocess
import sys 

def consensus_fetcher_prob_prog(k):
    consensus={}
    bw_exitguard={}
    bw_exit={}
    bw_guard={}
    bw_middle={}
    file =  ("/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/v3bw_{}").format(k)
    # file = conf.getpath('paths', 'v3bw_fname').format(k)
    otf=open(file, "rt")
    # otf=open("/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/v3bw", "rt")
    for line in otf:
        # print(line)
        if m := re.search('(\S+)(.*)bw=(\d+)(.*)nick=relay(\d+)(\S+)', line):
            if m.group(6)=='exitguard':
                bw_exitguard['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)),'relay': 'relay'+m.group(5)+m.group(6), 'bw': int(m.group(3)), 'fp': m.group(1)[9:]}
            if m.group(6)=='exit':
                bw_exit['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)),'relay': 'relay'+m.group(5)+m.group(6), 'bw': int(m.group(3)), 'fp': m.group(1)[9:]}
            if m.group(6)=='guard':
                bw_guard['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)),'relay': 'relay'+m.group(5)+m.group(6), 'bw': int(m.group(3)), 'fp': m.group(1)[9:]}
            if m.group(6)=='middle':
                bw_middle['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)),'relay': 'relay'+m.group(5)+m.group(6), 'bw': int(m.group(3)), 'fp': m.group(1)[9:]}
            consensus['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)), 'relay': 'relay'+m.group(5)+m.group(6), 'bw': int(m.group(3)), 'fp': m.group(1)[9:]}
    
    bw_exitguard = sorted(bw_exitguard.items(), key=lambda item: item[1]['number'])
    bw_exit = sorted(bw_exit.items(), key=lambda item: item[1]['number'])
    bw_guard = sorted(bw_guard.items(), key=lambda item: item[1]['number'])
    bw_middle = sorted(bw_middle.items(), key=lambda item: item[1]['number'])
    consensus = sorted(consensus.items(), key=lambda item: item[1]['number'])
    # print('consensus: '+str(consensus))
    return consensus, bw_exitguard, bw_exit, bw_guard, bw_middle

def observation_fetcher_prob_prog():
    consensus={}
    # file =  conf.getpath('paths', 'observation_file')
    file =  ("/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/observations.txt")
    otf=open(file, "rt")
    # otf=open("/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/v3bw", "rt")
    for line in otf:
        if m := re.search('(\S+)(.*)bw=(\d+)(.*)nick=relay(\d+)(\S+)', line):
            if 'relay'+m.group(5)+m.group(6) in consensus:
                consensus['relay'+m.group(5)+m.group(6)]['bw'].append(max(0.001,int(m.group(3))))
            else:
                consensus['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)),'relay': 'relay'+m.group(5)+m.group(6), 'bw': [max(0.001,int(m.group(3)))], 'fp': m.group(1)[9:]}
    consensus = sorted(consensus.items(), key=lambda item: item[1]['number'])
    return consensus

def weights_fetcher_prob_prog(file):
    weight_used={}
    # file =  conf.getpath('paths', 'observation_file')
    otf=open(file, "rt")
    for line in otf:
        if m := re.search('(\S+)(.*)bw=(\d+)(.*)nick=relay(\d+)(\S+)', line):
            if 'relay'+m.group(5)+m.group(6) in weight_used:
                weight_used['relay'+m.group(5)+m.group(6)]['weight'].append(float(m.group(3)+m.group(4).replace("\t", "")))
            else:
                weight_used['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)), 'relay': 'relay'+m.group(5)+m.group(6), 'weight': [float(m.group(3)+m.group(4).replace("\t", ""))], 'fp': m.group(1)[9:]}
    weight_used = sorted(weight_used.items(), key=lambda item: item[1]['number'])
    # print(weight_used)
    return weight_used

def parameters_fetcher_prob_prog(file):
    parameters={}
    # file =  conf.getpath('paths', 'observation_file')
    otf=open(file, "rt")
    for line in otf:
        if m := re.search('(\S+)(.*)par=(\d+)(.*)nick=relay(\d+)(\S+)', line):
            if 'relay'+m.group(5)+m.group(6) in parameters:
                parameters['relay'+m.group(5)+m.group(6)]['par'].append(float(m.group(3)+m.group(4).replace("\t", "")))
            else:
                parameters['relay'+m.group(5)+m.group(6)]={'number': int(m.group(5)), 'relay': 'relay'+m.group(5)+m.group(6), 'par': [float(m.group(3)+m.group(4).replace("\t", ""))], 'fp': m.group(1)[9:]}
    parameters = sorted(parameters.items(), key=lambda item: item[1]['number'])
    # print(parameters)
    return parameters




# import numbers
class DoubleAffineTransform(numpyro.distributions.transforms.Transform):
    """
#     Transform via the pointwise affine mapping :math:`y = (\text{loc1} + \text{scale1} \times x)/(\text{loc2} + \text{scale2} \times x)`.

#     Args:
#         loc1 / loc2 (Tensor or float): Location parameter.
#         scale1 / scale2 (Tensor or float): Scale parameter.
#         event_dim (int): Optional size of `event_shape`. This should be zero
#             for univariate random variables, 1 for distributions over vectors,
#             2 for distributions over matrices, etc.
#     """
  
    bijective = True
    
    def __init__(self, loc1, scale1, loc2, scale2, total_nb_paths,obs, event_dim=0):

        self.loc1 = loc1
        self.loc2 = loc2
        self.scale1 = scale1
        self.scale2 = scale2
        self.total_nb_paths = total_nb_paths
        self.obs = obs
        self._event_dim = event_dim

    @property
    def domain(self):
        return constraints._Interval(0, self.total_nb_paths)


    @property
    def codomain(self):
        return constraints._GreaterThan(self.obs)

    def _call(self, x):
        scale1 = self.scale1
        scale2 = self.scale2
        loc1 = self.loc1
        loc2 = self.loc2
        total_nb_paths = self.total_nb_paths
        result = (loc1 + scale1 * x) / (loc2 + scale2 * x)
        return result


    def _inverse(self, y):
        scale1 = self.scale1
        scale2 = self.scale2
        loc1 = self.loc1
        loc2 = self.loc2
        total_nb_paths = self.total_nb_paths
        
        result = (loc2 * y - loc1) / (scale1 - scale2 * y)
        return result

    
    def log_abs_det_jacobian(self, x, y, intermediates=None):

        
        scale1 = self.scale1
        scale2 = self.scale2
        loc1 = self.loc1
        loc2 = self.loc2
        obs = self.obs
        total_nb_paths= self.total_nb_paths
        result = jnp.broadcast_to(jnp.log(jnp.abs((scale1  - scale2 * y)/ (loc2 + scale2 * x))), jnp.shape(x))
        return result
        
def nonzero(a):
    return jax.lax.cond(a[0] > 0, a[0], lambda _: 1, a[0], lambda _: 0)

def model(guess_scale,  sum_bottleneck_weight_gc, sum_bottleneck_weight_mc, sum_bottleneck_weight_ec, sum_weight_gc, sum_weight_mc, sum_weight_ec, number_of_paths, used_weight_guard, used_weight_middle,  used_weight_exit, data_of_relay):
    scale0 = guess_scale 
    shape0 = 1
    C=numpyro.sample("capacity", dist.Weibull(scale0, shape0))    
    with numpyro.plate("obs", (data_of_relay).size) as i:
        transform2 = [DoubleAffineTransform(loc1=C+number_of_paths*used_weight_guard[i]*sum_bottleneck_weight_mc[i]+number_of_paths*used_weight_guard[i]*sum_bottleneck_weight_ec[i]+number_of_paths*used_weight_middle[i]*sum_bottleneck_weight_gc[i]+number_of_paths*used_weight_middle[i]*sum_bottleneck_weight_ec[i]+number_of_paths*used_weight_exit[i]*sum_bottleneck_weight_gc[i]+number_of_paths*used_weight_exit[i]*sum_bottleneck_weight_mc[i], scale1=-sum_bottleneck_weight_gc[i]-sum_bottleneck_weight_mc[i]-sum_bottleneck_weight_ec[i], loc2=1-number_of_paths*used_weight_middle[i]*nonzero(used_weight_guard[i])*(1-sum_weight_gc[i])-number_of_paths*used_weight_middle[i]*nonzero(used_weight_exit[i])*(1-sum_weight_ec[i])-number_of_paths*used_weight_guard[i]*nonzero(used_weight_middle[i])*(1-sum_weight_mc[i])-number_of_paths*used_weight_guard[i]*nonzero(used_weight_exit[i])*(1-sum_weight_ec[i])-number_of_paths*used_weight_exit[i]*nonzero(used_weight_guard[i])*(1-sum_weight_gc[i])-number_of_paths*used_weight_exit[i]*nonzero(used_weight_middle[i])*(1-sum_weight_mc[i]), scale2=nonzero(used_weight_guard[i])*(1-sum_weight_gc[i])+nonzero(used_weight_middle[i])*(1-sum_weight_mc[i])+nonzero(used_weight_exit[i])*(1-sum_weight_ec[i]), total_nb_paths=number_of_paths, obs=data_of_relay[i] )]
        return numpyro.sample("obs_{}".format(i), dist.TransformedDistribution(dist.Poisson(number_of_paths*(used_weight_guard[i]+used_weight_middle[i]+used_weight_exit[i])), transform2), obs=data_of_relay[i])
        
    
def guide(guess_scale,  sum_bottleneck_weight_gc, sum_bottleneck_weight_mc, sum_bottleneck_weight_ec, sum_weight_gc, sum_weight_mc, sum_weight_ec, number_of_paths, used_weight_guard, used_weight_middle,  used_weight_exit, data_of_relay):
    transform = [dist.transforms.AffineTransform(loc=jax.numpy.max(data_of_relay), scale=1)]
    scale_q = numpyro.param("scale_q", guess_scale, constraint=constraints.positive)
    shape_q = numpyro.param("shape_q", 1, constraint=constraints.positive)
    return numpyro.sample("capacity", dist.TransformedDistribution(dist.Weibull(scale_q, shape_q), transform))

def probabilistic_programming_algo(mdl, gd, num_steps_inference, relay_estimate, relay_par1gc, relay_par1mc, relay_par1ec, relay_par2gc, relay_par2mc, relay_par2ec, number_of_paths, relay_weight_first, relay_weight_second, relay_weight_third, relay_observations):    
    optimizer =Adam(0.01)
    svi = SVI(mdl, gd, optim=optimizer, loss=Trace_ELBO())
    svi_result= svi.run(jax.random.PRNGKey(0), num_steps_inference, relay_estimate,  jnp.array(relay_par1gc), jnp.array(relay_par1mc), jnp.array(relay_par1ec), jnp.array(relay_par2gc), jnp.array(relay_par2mc), jnp.array(relay_par2ec),number_of_paths, jnp.array(relay_weight_first), jnp.array(relay_weight_second), jnp.array(relay_weight_third),jnp.array(relay_observations), progress_bar=False)
    # global result
    # result = max(relay_observations)+ svi_result.params["scale_q"]
    return max(relay_observations)+ svi_result.params["scale_q"]


### THREADING #############
# import threading
# result = None

# def main():
# #     for i in range(3):
#     thread = threading.Thread(target=probabilistic_programming_algo(model, guide, 100000, 10750, [0], [0], [1796.45], [0], [0], [1], 100, [0], [0], [0.25], [6636]))
#     thread.start()

#     # wait here for the result to be available before continuing
#     thread.join()
#     print('The result is', result)

# if __name__ == '__main__':
#     main()

## POOL ########

# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:
#         res = pool.apply_async(probabilistic_programming_algo, (model, guide, 100000, 10750, [0], [0], [1796.45], [0], [0], [1], 100, [0], [0], [0.25], [6636],))      # runs in *only* one process
#         print(res.get(timeout=2))
#         # time.sleep(2)
#         # print(res.get(timeout=0))

## SLEEP ####
if __name__ == '__main__':
    k = 0
    with subprocess.Popen(["/usr/bin/tail", "-f", '/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/observations.txt'], 
        stdout=subprocess.PIPE, text=True) as tail:
        for line in tail.stdout:
            print("Got line: ", line.strip())
            if "End epoch" in line:
                output = subprocess.getoutput("pidof shadow")
                print(output)
                # print("kill -SIGSTOP "+str(output))
                os.system("kill -STOP "+str(output))
                print('killed shadow')
                estimates, estimates_exitguard, estimates_exit, estimates_guard, estimates_middle = consensus_fetcher_prob_prog(k)
                print('estimates: '+str(estimates))

                est=np.array([])
                for d in estimates:
                    # print('part of estimate: '+str(d[1]['bw']))
                    est = np.append(est, [d[1]['bw']])
                    # est.append(d[1]['bw'])
                print('est: '+str(est))

                file =  "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_first"
                weight_used_first = weights_fetcher_prob_prog(file)
                # print('weight_used_first: '+str(weight_used_first))

                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_second"
                weight_used_second = weights_fetcher_prob_prog(file)
                # print('weight_used_second: '+str(weight_used_second))

                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_third"
                weight_used_third = weights_fetcher_prob_prog(file)
                # print('weight_used_third: '+str(weight_used_third))

                w_1 =[np.array([]) for i in range(len(est))]
                for i in range(len(weight_used_first)):
                # for d in weight_used_first:
                    w_1[i] = np.append(w_1[i], [weight_used_first[i][1]['weight']])
                    # w_1.append(d[1]['weight'])
                print('w_1: '+str(w_1))

                w_2 =[np.array([]) for i in range(len(est))]
                for i in range(len(weight_used_second)):
                # for d in weight_used_second:
                    w_2[i] = np.append(w_2[i], [weight_used_second[i][1]['weight']])
                    # w_2.append(d[1]['weight'])
                print('w_2: '+str(w_2))

                w_3 =[np.array([]) for i in range(len(est))]
                for i in range(len(weight_used_third)):
                # for d in weight_used_third:
                    w_3[i]= np.append(w_3[i], [weight_used_third[i][1]['weight']])
                    # w_3.append(d[1]['weight'])
                print('w_3: '+str(w_3))

                observations = observation_fetcher_prob_prog()
                # print('observations: '+str(observations))

                obs=[np.array([]) for i in range(len(est))]
                for i in range(len(observations)):
                # for d in observations:
                    obs[i] = np.append(obs[i], [observations[i][1]['bw']])
                print(obs)

                
                #get them from file
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1gc"
                par1gc_dict = parameters_fetcher_prob_prog(file)
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1mc"
                par1mc_dict = parameters_fetcher_prob_prog(file)
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1ec"
                par1ec_dict = parameters_fetcher_prob_prog(file)
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2gc"
                par2gc_dict = parameters_fetcher_prob_prog(file)
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2mc"
                par2mc_dict = parameters_fetcher_prob_prog(file)
                file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2ec"
                par2ec_dict = parameters_fetcher_prob_prog(file)
                par1gc =[np.array([]) for i in range(len(est))]
                for i in range(len(par1gc_dict)):
                    par1gc[i] = np.append(par1gc[i], [par1gc_dict[i][1]['par']])
                        # par1gc.append(d[1]['par'])
                par1mc =[np.array([]) for i in range(len(est))]
                for i in range(len(par1mc_dict)):
                    par1mc[i] = np.append(par1mc[i], [par1mc_dict[i][1]['par']])
                par1ec =[np.array([]) for i in range(len(est))]
                for i in range(len(par1ec_dict)):
                    par1ec[i] = np.append(par1ec[i], [par1ec_dict[i][1]['par']])
                par2gc =[np.array([]) for i in range(len(est))]
                for i in range(len(par2gc_dict)):
                    par2gc[i] = np.append(par2gc[i], [par2gc_dict[i][1]['par']])
                par2mc =[np.array([]) for i in range(len(est))]
                for i in range(len(par2mc_dict)):
                    par2mc[i] = np.append(par2mc[i], [par2mc_dict[i][1]['par']])
                par2ec =[np.array([]) for i in range(len(est))]
                for i in range(len(par2ec_dict)):
                    par2ec[i] = np.append(par2ec[i], [par2ec_dict[i][1]['par']])
                print(par1gc)
                print(par1mc)
                print(par1ec)
                print(par2gc)
                print(par2mc)
                print(par2ec)    

                new_est = copy.deepcopy(est)
                for i in range(len(new_est)):
                    print(est[i], par1gc[i], par1mc[i], par1ec[i], par2gc[i], par2mc[i], par2ec[i],w_1[i], w_2[i], w_3[i], obs[i])
                    new_est[i] = probabilistic_programming_algo(model, guide, 100000, est[i], par1gc[i], par1mc[i], par1ec[i], par2gc[i], par2mc[i], par2ec[i], 100, w_1[i], w_2[i], w_3[i], obs[i])

                # estimate = probabilistic_programming_algo(model, guide, 100000, 10750, [0], [0], [1796.45], [0], [0], [1], 100, [0], [0], [0.25], [6636])
                output2 = ('/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/v3bw_{}').format(k+1)
                out_dir = os.path.dirname(output2)
                with open(output2, 'a') as fd:
                    i = 0
                    for  d in estimates:
                        fd.write('node_id='+str(d[1]['fp'])+'\t'+'bw='+str(int(new_est[i]))+'\t'+'nick='+str(d[1]['relay'])+'\n')
                        i = i + 1
                k = k+1
                os.system("kill -CONT "+str(output))
                print('Relaunched shadow')
            if k==5: ##nb of epochs
                break
    ## read the obs file and see the tail _ epoch number
    ## once epoch finish = 1) pause shadow process
    ## run below
    
    
    # for k in range(5):
    #     estimates, estimates_exitguard, estimates_exit, estimates_guard, estimates_middle = consensus_fetcher_prob_prog(k)
    #     # print('estimates: '+str(estimates))

    #     est=np.array([])
    #     for d in estimates:
    #         # print('part of estimate: '+str(d[1]['bw']))
    #         est = np.append(est, [d[1]['bw']])
    #         # est.append(d[1]['bw'])
    #     print('est: '+str(est))

    #     file =  "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_first"
    #     weight_used_first = weights_fetcher_prob_prog(file)
    #     # print('weight_used_first: '+str(weight_used_first))

    #     file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_second"
    #     weight_used_second = weights_fetcher_prob_prog(file)
    #     # print('weight_used_second: '+str(weight_used_second))

    #     file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/weight_used_third"
    #     weight_used_third = weights_fetcher_prob_prog(file)
    #     # print('weight_used_third: '+str(weight_used_third))

    #     w_1 =[np.array([]) for i in range(len(est))]
    #     for i in range(len(weight_used_first)):
    #     # for d in weight_used_first:
    #         w_1[i] = np.append(w_1[i], [weight_used_first[i][1]['weight']])
    #         # w_1.append(d[1]['weight'])
    #     print('w_1: '+str(w_1))

    #     w_2 =[np.array([]) for i in range(len(est))]
    #     for i in range(len(weight_used_second)):
    #     # for d in weight_used_second:
    #         w_2[i] = np.append(w_2[i], [weight_used_second[i][1]['weight']])
    #         # w_2.append(d[1]['weight'])
    #     print('w_2: '+str(w_2))

    #     w_3 =[np.array([]) for i in range(len(est))]
    #     for i in range(len(weight_used_third)):
    #     # for d in weight_used_third:
    #         w_3[i]= np.append(w_3[i], [weight_used_third[i][1]['weight']])
    #         # w_3.append(d[1]['weight'])
    #     print('w_3: '+str(w_3))

    #     observations = observation_fetcher_prob_prog()
    #     # print('observations: '+str(observations))

    #     obs=[np.array([]) for i in range(len(est))]
    #     for i in range(len(observations)):
    #     # for d in observations:
    #         obs[i] = np.append(obs[i], [observations[i][1]['bw']])
    #     print(obs)

    #     if k ==0:
    #         par1gc = [np.array([]) for i in range(len(est))]
    #         par1mc = [np.array([]) for i in range(len(est))]
    #         par1ec = [np.array([]) for i in range(len(est))]
    #         par2gc = [np.array([]) for i in range(len(est))]
    #         par2mc = [np.array([]) for i in range(len(est))]
    #         par2ec = [np.array([]) for i in range(len(est))]
    #     else:
    #         #get them from file
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1gc"
    #         par1gc_dict = parameters_fetcher_prob_prog(file)
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1mc"
    #         par1mc_dict = parameters_fetcher_prob_prog(file)
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par1ec"
    #         par1ec_dict = parameters_fetcher_prob_prog(file)
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2gc"
    #         par2gc_dict = parameters_fetcher_prob_prog(file)
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2mc"
    #         par2mc_dict = parameters_fetcher_prob_prog(file)
    #         file = "/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/par2ec"
    #         par2ec_dict = parameters_fetcher_prob_prog(file)
    #         par1gc =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par1gc_dict)):
    #             par1gc[i] = np.append(par1gc[i], [par1gc_dict[i][1]['par']])
    #             # par1gc.append(d[1]['par'])
    #         par1mc =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par1mc_dict)):
    #             par1mc[i] = np.append(par1mc[i], [par1mc_dict[i][1]['par']])
    #         par1ec =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par1ec_dict)):
    #             par1ec[i] = np.append(par1ec[i], [par1ec_dict[i][1]['par']])
    #         par2gc =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par2gc_dict)):
    #             par2gc[i] = np.append(par2gc[i], [par2gc_dict[i][1]['par']])
    #         par2mc =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par2mc_dict)):
    #             par2mc[i] = np.append(par2mc[i], [par2mc_dict[i][1]['par']])
    #         par2ec =[np.array([]) for i in range(len(est))]
    #         for i in range(len(par2ec_dict)):
    #             par2ec[i] = np.append(par2ec[i], [par2ec_dict[i][1]['par']])
            
    #     new_est = copy.deepcopy(est)
    #     for i in range(len(new_est)):
    #         print(est[i], par1gc[i], par1mc[i], par1ec[i], par2gc[i], par2mc[i], par2ec[i],w_1[i], w_2[i], w_3[i], obs[i])
    #         new_est[i] = probabilistic_programming_algo(model, guide, 100000, est[i], par1gc[i], par1mc[i], par1ec[i], par2gc[i], par2mc[i], par2ec[i], 100, w_1[i], w_2[i], w_3[i], obs[i])

    #     # estimate = probabilistic_programming_algo(model, guide, 100000, 10750, [0], [0], [1796.45], [0], [0], [1], 100, [0], [0], [0.25], [6636])
    #     output = ('/home/hdarir2/simulation_final_12relays/shadow.data/hosts/bwauthority/v3bw_{}').format(k+1)
    #     out_dir = os.path.dirname(output)
    #     with open(output, 'a') as fd:
    #         i = 0
    #         for  d in estimates:
    #             fd.write('node_id='+str(d[1]['fp'])+'\t'+'bw='+str(new_est[i])+'\t'+'nick='+str(d[1]['relay'])+'\n')
    #             i = i + 1
    ##  CONTINUE SHADOW SIM          
    
