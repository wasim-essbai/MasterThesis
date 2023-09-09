import numpy as np


def linear_tolerance(y, maxAcc, th, is_acc):
    if is_acc:
        return np.maximum(np.minimum(y, maxAcc) - th, 0) / (maxAcc - th)
    else:
        return np.maximum(th - np.maximum(y, maxAcc), 0) / (th - maxAcc)


def linear_dist(x, ua, la):
    result = []
    for i in x:
        if la <= i <= ua:
            result.append((2 / ((ua - la) ** 2)) * (ua - i))
        else:
            result.append(0)
    return np.array(result)
	
def uniform_dist(x, ua, la):
    result = []
    for i in x:
        if la <= i <= ua:
            result.append(1 / (ua - la))
        else:
            result.append(0)
    return np.array(result)

