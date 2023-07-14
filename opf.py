import numpy as np
import random 
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pypower.api import case9
from runpf import runpf2

## Note - power_array refers to a row of the x position matrix - a set of power values ([Pg1, Pg2...Qg3]).
##        This program is specified for an IEEE 9 bus system, some modifications may have to be made for general/larger cases.


#number of particles
p = 20

#number of dimensions/generators
d = 3
busses = 9

# initial cognitive, social factors
c1 = 2
c2 = 2

#number of iterations 
num_iters = 150

#inertia weight
w = 0.9

# some arbitrary boundary
bound = 1000

######### variable constraints

# (voltage generated in per units)
v_g_min = 0.9
v_g_max = 1.1

# (real power in MW)
p_g_min = 50
p_g_max = 200

# (reactive power in MVAr)
q_g_min = -20
q_g_max = 30

# tap ratio
t_min = 0.2
t_max = 1

# (power capacity in MVAr)
q_c_min = -20
q_c_max = 100

# (load voltage in per units)
v_l_min = 0.7
v_l_max = 1.2

# (apparent power in MVA)
s_l_min = 0
s_l_max = 500

# lambda constants
lam_p = 1
lam_v = 1
lam_q = 1
lam_s = 1

# lim
q_lim = 1
v_lim = 1
p_lim = 1

# initial cost?
j = 10


#initializing position/real power matrix
shape = (p, d)

x = np.random.uniform(p_g_min, p_g_max, size=shape)

# load analysis function ----------------------------------------------------------------------------------------------------------------------

def load_analysis(power_matrix, voltage_matrix, reactive_matrix, line_apparent_matrix):

    # voltage = []
    # reactive_power = []
    # line_apparent_power = []

    ppc = case9()

    for i in range(np.shape(power_matrix)[0]):

        ppc['gen'][0:3, 1] = power_matrix[i][:]

        results = runpf2(ppc)

        # may have to assign more explicitly, check later
        voltage_matrix[i] = results[0]['bus'][:, 7]
        reactive_matrix[i] = results[0]['gen'][:, 2]
        line_apparent_matrix[i] = (results[0]['branch'][:, 15]**2 + results[0]['branch'][:, 16]**2)**0.5

        # voltage_matrix.append(results[0]['bus'][:, 7])
        # reactive_matrix.append(results[0]['gen'][:, 2])
        # line_apparent_matrix.append((results[0]['branch'][:, 15]**2 + results[0]['branch'][:, 16]**2)**0.5)
    
    # print('length of matrices:', np.shape(voltage_matrix), np.shape(reactive_matrix), np.shape(line_apparent_matrix))

    # note: voltage and reactive_power are matrices of shape (num particles, num busses), (num particles, num generators), respectively
    # note: line_apparent_power is a matrix of shape (num particles, num busses)
    return voltage_matrix, reactive_matrix, line_apparent_matrix


# initializing velocity matrix ----------------------------------------------------------------------------------------------------------------
v  = np.random.uniform(-1, 1, shape)


# initializing personal best matrix, global best array
pbest = np.zeros((p, 2*d + 2*busses))
pbest[:, :d] = x
# initializing pbest to have not just a 
pbest[:, 2*d : 2*d + busses], pbest[:, d:2*d], pbest[:, 2*d + busses : 2*(d+busses)] = load_analysis(x, np.empty((p, busses)), np.empty((p, d)), np.empty((p, busses)))

gbest = pbest[0]

qbest = q_g_max


# rest of the functions ------------------------------------------------------------------------------------------------------------------------

def penalty_schema(power_array, reactive_array, voltage_gen_array, voltage_load_array, line_apparent_power):

    penalty = 0

    # assessing penalties for generators
    for i in range(len(reactive_array)):
        
        # real power
        if not(p_g_min <= power_array[i] <= p_g_max):

            if power_array[i] > p_g_max:
                penalty += (power_array[i] - p_g_max)**2

            elif power_array[i] < p_g_min:
                penalty += (p_g_min - power_array[i])**2

            else:
                print('error with penalty schema, real power gen.')
        
        # reactive power
        if not(q_g_min <= reactive_array[i] <= q_g_max):

            if reactive_array[i] > q_g_max:
                penalty += (reactive_array[i] - q_g_max)**2

            elif reactive_array[i] < q_g_min:
                penalty += (q_g_min - reactive_array[i])**2

            else:
                print('error with penalty schema, reactive power gen.')

        # voltage
        if not(v_g_min <= voltage_gen_array[i] <= v_g_max):

            if voltage_gen_array[i] > v_g_max:
                penalty += (voltage_gen_array[i] - v_g_max)**2

            elif voltage_gen_array[i] < v_g_min:
                penalty += (v_g_min - voltage_gen_array[i])**2

            else:
                print('error with penalty schema, voltage gen.')

        # Qc?
    
    # loads
    for j in range(len(voltage_load_array)):

        # voltage
        if not(v_l_min <= voltage_load_array[j] <= v_l_max):

            if voltage_load_array[j] > v_l_max:
                penalty += (voltage_load_array[j] - v_l_max)**2

            elif voltage_load_array[j] < v_g_min:
                penalty += (v_l_min - voltage_load_array[j])**2

            else:
                print('error with penalty schema, reactive power gen.')
    
    # branches
    for k in range(len(line_apparent_power)):
        # apparent power
        if line_apparent_power[k] > s_l_max:

            penalty += line_apparent_power[k] - s_l_max
    
    return penalty
        


#note: compute_fcn computes the cost function for a single row of the matrices
def compute_fcn(j, power_array, reactive_array, voltage_array, branch_load_array):


    # real power component
    p_g_sum = 0

    for i in range(len(power_array)):

        p_g_sum += (power_array[i]- p_lim)**2
    
    p_g_sum *= lam_p


    # voltage component
    v_sum = 0

    for i in range(len(voltage_array)):
        # if voltage_array[i] != 0:
        v_sum += (voltage_array[i] - v_lim)**2
    
    v_sum *= lam_v

    # reactive power component
    q_sum = 0

    for i in range(len(reactive_array)):
        # if reactive_array[i] != 0:
        q_sum += (reactive_array[i] - q_lim)**2
    
    q_sum *= lam_q

    # reactive power component - do I use amount injected or amount delivered
    s_sum = 0

    for i in range(len(branch_load_array)):
        # if (branch_load_array[i][0] != 0) or (branch_load_array[i][1] != 0):
        s_sum += (branch_load_array[i] - s_l_max)**2
    
    s_sum *= lam_s

    j_aug = j + p_g_sum + v_sum + q_sum + s_sum

    penalty = penalty_schema(power_array, reactive_array, voltage_array[0:3], [voltage_array[4], voltage_array[6], voltage_array[8]], branch_load_array)

    j_aug += penalty

    return j_aug



# constraint function, returns boolean values based on location of particle
def constraint(voltage_array, real_power_array, reactive_power_array):

    v_g = voltage_array[0:3]
    v_l = [voltage_array[5], voltage_array[7],voltage_array[9]]

    # checking generator constraints
    for i in range(len(v_g)):
        if not (v_g_min <= v_g[i] <= v_g_max):
            return False
        
        elif not (p_g_min <= real_power_array[i] <= p_g_max):
            return False
        
        elif not (q_g_min <= reactive_power_array[i] <= q_g_max):
            return False
    
    # checking load constraints
    for j in range(len(v_l)):
        if not(v_l_min <= v_l[j] <= v_l_max):
            return False
        
    return True



def update_velocity(power_matrix, v, pbest, gbest):
    
    r1 = random.random()
    r2 = random.random()

    for i in range(np.shape(power_matrix)[0]):
    
        v[i] = (w*v[i]) + (c1*r1*(pbest[i, :d] - power_matrix[i])) + (c2*r2*(gbest[d] - power_matrix[i]))
        
    return v


def update_position(x, v, num_dimensions):

    for i in range(np.shape(x)[0]):

        for j in range(num_dimensions):
    
            # the array was index in this form so that individual bounds could be checked 
            x[i][j] = x[i][j] + v[i][j]

            if x[i][j] < p_g_min:
                x[i][j] = p_g_min
            
            # if x[i][j] > p_g_max:
            #     x[i][j] = p_g_max
        
    return x

def update_bests(pbest, gbest, power_matrix, voltage_matrix, reactive_matrix, branch_load_matrix, index):


    # np.shape(x)[0] accesses the first value that the np.shape() function delivers, which is the m value of the position matrix
    for i in range(np.shape(power_matrix)[0]):

        # constraint() will return a boolean value, which returns true if the particle is within the acceptable constraints
        # if constraint(x[i]) == True:


        if compute_fcn(j, power_matrix[i], reactive_matrix[i], voltage_matrix[i], branch_load_matrix[i]) < compute_fcn(j, pbest[i, :d],  pbest[i, d:2*d], pbest[i, 2*d : 2*d + busses], pbest[i, 2*d + busses : 2*(d+busses)]):

            pbest[i, :d],  pbest[i, 2*d : 2*d + busses], pbest[i, d:2*d], pbest[i, 2*d + busses : 2*(d+busses)] = power_matrix[i], voltage_matrix[i], reactive_matrix[i], branch_load_matrix[i]
            
        if compute_fcn(j, power_matrix[i], reactive_matrix[i], voltage_matrix[i], branch_load_matrix[i]) < compute_fcn(j, gbest[:d],  gbest[d:2*d], gbest[2*d : 2*d + busses], gbest[2*d + busses : 2*(d+busses)]):

            gbest[:d],  gbest[2*d : 2*d + busses], gbest[d:2*d],  gbest[2*d + busses : 2*(d+busses)] = power_matrix[i], voltage_matrix[i], reactive_matrix[i], branch_load_matrix[i]
            index = i


def optimize(x, v, pbest, gbest, j):

    voltage_matrix = np.empty((p, busses))
    reactive_matrix = np.empty((p, d))
    complex_line_power = np.empty((p, busses))
    # voltage_matrix = []
    # reactive_matrix = []
    # complex_line_power = []
    index = 0

    # initializing a gbest - should fix to work with constraint
    for i in range(np.shape(x)[0]):

        # voltage_matrix, reactive_matrix, complex_line_power = load_analysis(x, [], [], [])

        ##fix gbest
        if (compute_fcn(j, pbest[i,:d], pbest[i, d:2*d], pbest[i, 2*d : 2*d + busses], pbest[i, 2*d + busses : 2*(d+busses)]) < compute_fcn(j, gbest[:d],  gbest[d:2*d], gbest[2*d : 2*d + busses], gbest[2*d + busses : 2*(d+busses)])):
            
            gbest = pbest[i]
            index = i
        
        # else:
        #     # arbitrary, will need to be changed
        #     gbest = []
        #     for i in range(d):
        #         gbest.append(p_g_max)
        #     for i in range(d):
        #         gbest.append(q_g_max)
            
        #     for j in range(busses):
        #         gbest.append(v_lim)
            
        #     for j in range(busses):
        #         gbest.append(s_l_max)


    decrease = 0.5/num_iters
 
    j_list = []

    fig, ax = plt.subplots()

    
    for j in range(num_iters):

        print('iteration number:', j, '----------------------')

        global w 

        # decreasing inertia over time, decreasing exploration, increasing exploitation
        w -= decrease

        v_new = update_velocity(x, v, pbest, gbest)

        x = update_position(x, v_new, d)

        load_analysis(x, voltage_matrix, reactive_matrix, complex_line_power)

        update_bests(pbest, gbest, x, voltage_matrix, reactive_matrix, complex_line_power, index)

        j_list.append(compute_fcn(j, gbest[:d],  gbest[d:2*d], gbest[2*d : 2*d + busses], gbest[2*d + busses : 2*(d+busses)]))

        ax.cla()
        ax.plot(range(len(j_list)), j_list)

        ax.set_xlabel("Time")
        ax.set_ylabel("Global Best Cost")
        ax.set_title("Global Best Cost vs. Time")
        plt.pause(0.001)

    
    j_iterations = range(num_iters + 1)  # +1 to include the initial value


    print(j_list)
    print(min(j_list))

    
    return gbest

# calling function, testing -------------------------------------------------------------------------------------------------------------------

val = optimize(x, v, pbest, gbest, j=10)

print(val)
print(compute_fcn(10, val[:d], val[d:2*d], val[2*d:2*d + busses], val[2*d + busses:2*(d+busses)]))
print(compute_fcn(10, [89.8, 134.2, 94], [12.95, 0.05, -22.6], [1.1, 1.097, 1.087, 1.094, 1.084, 1.1, 1.089, 1.1, 1.072], [90.25, 37.3, 60.2, 97.88, 42.35, 62.2, 134.3, 72.5, 56]))

# print(load_analysis(gbest[:d], [], [], []))

plt.show()


