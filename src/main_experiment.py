import numpy as np
from common import *
import itertools
import requests
import json

script_name = __file__

def add_jobs(admin_url, admin_id, jobs, password, timeout):
    try:
        payload =dict(
          jobs=jobs,
          administrator_id=admin_id,
          timeout=timeout,
          mode='populate',
          password=password)

        url = "http://%s/add" % (admin_url)
        headers = {'content-type': 'application/json'}
        return requests.post(url, data=json.dumps(payload), headers=headers)
    except Exception,e:
        return str(e)

def gen_jobs_for_question(question, num):
    return [{"question": question} for i in range(num)]

def gen_jobs_for_all_questions(num):
    question = ["Imagine you woke up with an extra opposable thumb on each hand. List benefits and drawbacks.",
                "List unusual uses for a mop.",
                "I would like to lose weight. Please list things a person can do to lose weight."]

    jobs = [gen_jobs_for_question(q, num) for q in question]
    return list(itertools.chain.from_iterable(jobs))

if __name__=='__main__':
    exp_location, schema, expid, tc, admin_url = initialize_from_cmd(script_name)

    # Upload jobs hack
    # This will re-add jobs every time, which is awful

    jobs = gen_jobs_for_all_questions(10)

    with sl.Experiment(exp_location, schema, expid) as exp:
        fives_aid = "%s_fives" % (expid)
        r = add_jobs(admin_url, fives_aid, jobs, 'chicoritastranglelemon', timeout=700)
        print r
        fives_key = start_trial(tc, RandomStorm(5, 10, admin_url, fives_aid), exp,'fives')

        tens_aid = "%s_tens" % (expid)
        r = add_jobs(admin_url, tens_aid, jobs, 'chicoritastranglelemon', timeout=1000)
        print r
        tens_key = start_trial(tc, RandomStorm(10, 10, admin_url, tens_aid), exp,'tens')

        twenties_aid = "%s_twenties" % (expid)
        r = add_jobs(admin_url, twenties_aid, jobs, 'chicoritastranglelemon', timeout=1600)
        print r
        twenties_key = start_trial(tc, RandomStorm(20, 10, admin_url, twenties_aid), exp,'twenties')

        fives_results = attempt_finish_trial(tc, exp, fives_key)
        tens_results = attempt_finish_trial(tc, exp, tens_key)
        twenties_results = attempt_finish_trial(tc, exp, twenties_key)

