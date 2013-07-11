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
    n = "as many as possible" if n is None else "%i" % (n)

    charity_question = """The Electronic Frontier Foundation (EFF, https://www.eff.org)
      is a nonprofit whose goal is to protect individual and consumer rights with respect
      to digital technologies. For example, EFF filed and settled a class-action lawsuit
      against Sony after Sony sold music CDs that installed software on a person's computer
      that prevented them from copying CDs. It is now bringing a lawsuit against the US
      government to limit the degree to which it spies on its citizens through secret NSA
      programs.\n\nWhile EFF is doing important work, it is relatively unknown to the
      public, and relies on donations from the public. Brainstorm %s ways to raise funds
      for the EFF, where the fundraising also increases awareness of the need to protect
      digital rights."""

    turk_question = """Mechanical Turk currently lacks a dedicated interface for
      smartphones (iPhone, Androids, etc.) and tablets (e.g., the iPad). While some HITs
      may be more difficult to perform on a mobile device (e.g., those that require lots of
      typing), some may be easier. For example, the unique features of mobile devices
      (their extreme mobility, built-in cameras, multi-touch interfaces, built-in GPS, built-in
      microphones, ability to call others) may make it possible to perform some tasks that are
      impossible or extremely difficult with normal computers.\n\nBrainstorm %s different types
      of HITs that could be performed with mobile devices that cannot be performed with a
      regular computer, or which are much easier on mobile devices. Be as specific as possible
      in your ideas."""

    question = [charity_question % (n),
                turk_question % (n)]

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
                                  num_assignments_per_question * len(questions),
                                  reward,
                                  administrator_URL,
                                  administrator_id,
                                  num_responses is None),
                      exp,
                      HIT_id)
    return key

def curry_post_jobs_for_n_responses(administrator_URL, num_assignments_per_question, tc, exp):
    def post(administrator_id, HIT_id, questions, num_responses, reward):
        return post_jobs_for_n_responses(administrator_URL, administrator_id, HIT_id, questions,
            num_assignments_per_question, num_responses, reward, tc, exp)

    return post

if __name__=='__main__':
    exp_location, schema, expid, tc, admin_url = initialize_from_cmd(script_name)

    with sl.Experiment(exp_location, schema, expid) as exp:

        post = curry_post_jobs_for_n_responses(administrator_URL=admin_url,
                                               num_assignments_per_question=5,
                                               tc=tc,
                                               exp=exp)

        #fives_key = post("%s_fives" % (expid), 'fives', get_questions_for_n(5), 5, 0.18)
        #tens_key = post("%s_tens" % (expid), 'tens', get_questions_for_n(10), 10, 0.35)
        twenties_key = post("%s_twenties" % (expid), 'twenties', get_questions_for_n(20), 20, 0.75)
        #fifties_key = post("%s_fifties" % (expid), 'fifties', get_questions_for_n(50), 50, 1.75)
        #infini_key = post("%s_infini" % (expid), 'infini', get_questions_for_n(None), None, 1.75)

        #fives_results = attempt_finish_trial(tc, exp, fives_key)
        #tens_results = attempt_finish_trial(tc, exp, tens_key)
        twenties_results = attempt_finish_trial(tc, exp, twenties_key)
        #fifties_results = attempt_finish_trial(tc, exp, fifties_keys)

