import numpy as np
from common import *

script_name = __file__

if __name__=='__main__':
    exp_location, schema, expid, tc = initialize_from_cmd(script_name)

    with sl.Experiment(exp_location, schema, expid) as exp:
        fives_key = start_trial(tc, RandomStorm(5), exp,'fives')
        tens_key = start_trial(tc, RandomStorm(10), exp,'tens')
        twenties_key = start_trial(tc, RandomStorm(20), exp,'twenties')

        fives_results = attempt_finish_trial(tc, exp, fives_key)
        tens_results = attempt_finish_trial(tc, exp, tens_key)
        twenties_results = attempt_finish_trial(tc, exp, twenties_key)

