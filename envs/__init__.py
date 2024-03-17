from envs.scenario import *
from envs.environment import *

# def get_env(name):
#     if name.lower() == 'shunqing':
#         return shunqing
#     elif name.lower() == 'astlingen':
#         return astlingen
#     else:
#         raise AssertionError("Unknown environment %s"%str(name))
    
def get_env(name):
    try:
        return eval(name)
    except:
        raise AssertionError("Unknown environment %s"%str(name))