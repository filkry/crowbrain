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

def gen_jobs_for_all_questions(questions, num_assignments_per_question):
    jobs = [gen_jobs_for_question(q, num_assignments_per_question) for q in questions]
    return list(itertools.chain.from_iterable(jobs))


def get_questions_for_n(n):
    n = "as many as possible" is n is None else "%i" % (n)

    question = ["Please brainstorm %s ways that Mechanical Turk could be improved for workers. Be as specific as possible in your descriptions." % (n),
                "Please brainstorm %s different public events that could be used to raise money for Alzheimer's research. Be as specific as possible in your descriptions." % (n),
                "Many people have old iPods or MP3 players that they no longer use. Please brainstorm %s uses for old iPods/MP3 players. Assume that the devices' batteries no longer work, though they can be powered via external power sources. Also be aware that devices may *not* have displays. Be as specific as possible in your descriptions." % (n),
                "Imagine you are in a social setting and you have forgotten the name of somebody you know. Brainstorm %s ways you could learn their name without directly asking them. Be as specific as possible in your descriptions." % (n)]

    return question

def post_jobs_for_n_responses(administrator_URL, administrator_id, HIT_id, questions,
                              num_assignments_per_question, num_responses, reward, tc, exp):
    r = add_jobs(administrator_URL,
                 administrator_id,
                 gen_jobs_for_all_questions(questions, num_assignments_per_question),
                 'chicoritastranglelemon',
                 timeout=60*60*18 + 60)
    print r
    key = start_trial(tc,
                      RandomStorm(num_responses,
                                  num_assignments_per_question,
                                  reward,
                                  administrator_URL,
                                  administrator_id,
                                  num_responses is None),
                      exp,
                      HIT_id)
    return key

def curry_post_jobs_for_n_responses(administrator_URL, num_assignments_per_question, tc, exp):
    return lambda administrator_id, HIT_id, questions, num_responses, reward:
         return post_jobs_for_n_responses(administrator_URL, administrator_id, HIT_id, questions,
            num_assignments_per_question, num_responses, reward, tc, exp)

if __name__=='__main__':
    exp_location, schema, expid, tc, admin_url = initialize_from_cmd(script_name)

    with sl.Experiment(exp_location, schema, expid) as exp:

        post = curry_post_jobs_for_n_responses(administrator_URL=admin_url,
                                               num_assignments_per_question=5,
                                               tc=tc,
                                               exp=exp)

        fives_key = post("%s_fives" % (expid), 'fives', get_questions_for_n(5), 5, 0.18)
        tens_key = post("%s_tens" % (expid), 'tens', get_questions_for_n(10), 10, 0.35)
        twenties_key = post("%s_twenties" % (expid), 'twenties', get_questions_for_n(20), 20, 0.88)
        fifties_key = post("%s_fifties" % (expid), 'fifties', get_questions_for_n(50), 50, 1.75)
        infini_key = post("%s_infini" % (expid), 'infini', get_questions_for_n(None), None, 1.75)

        fives_results = attempt_finish_trial(tc, exp, fives_key)
        tens_results = attempt_finish_trial(tc, exp, tens_key)
        twenties_results = attempt_finish_trial(tc, exp, twenties_key)
        fifties_results = attempt_finish_trial(tc, exp, fifties_keys)

