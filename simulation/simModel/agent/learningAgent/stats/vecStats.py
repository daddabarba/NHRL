import numpy as np

def l(v):
    return np.linalg.norm(v)

def update_mean(mean, N, new_point):
    return (mean*N + new_point)*(1.0/(N+1))

def reshape_mean(mean):
    return np.append(mean, 0.0)

def update_stats(data, new_point):
    N_t_0 = data['N']

    mean_t_0 = data['mu']
    mean_t_1 = update_mean(mean_t_0, N_t_0, new_point)

    var_t_0 = data['sd']
    var_t_1 = (var_t_0*N_t_0 + l(new_point)**2 + N_t_0*(l(mean_t_0)**2) -(N_t_0+1)*(l(mean_t_1)**2))

    return {'mu': mean_t_1, 'sd': var_t_1, 'N': N_t_0+1}