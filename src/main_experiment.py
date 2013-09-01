import numpy as np
from common import *
import itertools
import requests
import json
import random

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
    job_type = None
    if 'EFF' in question:
      job_type = 'charity'
    elif 'Turk' in question:
      job_type = 'turk'
    elif 'MP3' in question:
      job_type = 'mp3'
    elif 'social' in question:
      job_type = 'forgot'

    return [{"question": question, "job_type": job_type} for i in range(num)]


def gen_jobs_for_all_questions_long(questions_assignments):
    jobs = []
    for question, num_assignments in questions_assignments:
      jobs.append(gen_jobs_for_question(question, num_assignments))
    return list(itertools.chain.from_iterable(jobs))

def gen_jobs_for_all_questions(questions, num_assignments_per_question):
    jobs = [gen_jobs_for_question(q, num_assignments_per_question) for q in questions]
    return list(itertools.chain.from_iterable(jobs))

def gen_jobs_random(questions, num_assignments):
    return [{"question": random.choice(questions)} for i in range(num_assignments)]

def get_questions_for_responses(n, append=False):
    n = "as many as possible" if append else "%i" % (n)

    charity_question = "<p>The Electronic Frontier Foundation (EFF) is a nonprofit whose goal is to protect individual rights with respect to digital and online technologies. For example, the EFF has initiated a lawsuit against the US government to limit the degree to which the US surveils its citizens via secret NSA programs. If you are unfamiliar with the EFF and its goals, read about it on its website (<a href=https://www.eff.org' target='_new'>https://www.eff.org</a>) or via other online sources (such as Wikipedia).</p>\
      <p>Brainstorm %s <em>new</em> ways the EFF can raise funds and simultaneously increase awareness. Your ideas <em>must be different from their current methods</em>, which include donation pages, merchandise, web badges and banners, affiliate programs with Amazon and eBay, and donating things such as airmiles, cars, or stocks. See the full list of their current methods here: <a href='https://www.eff.org/helpout' target='_new'>https://www.eff.org/helpout</a>. Be as specific as possible in your responses.</p>"

    turk_question = "<p>Mechanical Turk currently lacks a dedicated mobile app for performing HITs on smartphones (iPhone, Androids, etc.) or tablets (e.g., the iPad).</p>\
      <p>Brainstorm %s features for a mobile app to Mechanical Turk that would improve the worker's experience when performing HITs on mobile devices. Be as specific as possible in your responses.</p>"

    mp3_question = """<p>Many people have old iPods or MP3 players that they no longer use. Please brainstorm %s uses for old iPods/MP3 players. Assume that the devices' batteries no longer work, though they can be powered via external power sources. Also be aware that devices may <em>not</em> have displays. Be as specific as possible in your descriptions.</p>"""

    forgot_question = """<p>Imagine you are in a social setting and you have forgotten the name of somebody you know. Brainstorm %s ways you could learn their name without directly asking them. Be as specific as possible in your descriptions.</p>"""

    question = [charity_question % (n),
                turk_question % (n),
                mp3_question %(n),
                forgot_question %(n)]

    return question

### This part is what changes for the experiment, mainly

def get_num_participants(num_requested):
  if num_requested == 5:
    return [5, 5, 5, 5]
  elif num_requested == 10:
    return [5, 5, 5, 5]
  elif num_requested == 20:
    return [3, 0, 0, 0]
  elif num_requested == 50:
    return [6, 4, 5, 6]
  elif num_requested == 75:
    return [6, 5, 5, 5]
  elif num_requested == 100:
    return [7, 4, 5, 6]

def get_response_rewards(expid):
    return [
            (5, 0.18, "%s_fives" % expid, "fives", False),
            (10, 0.35, "%s_tens" % expid, "tens", False),
            (20, 0.70, "%s_twenties" % expid, "twenties", False),
            (50, 1.75, "%s_fifties" % expid, "fifties", False),
            (75, 2.65, "%s_seventy_fives" % expid, "seventy_fives", False),
            (100, 3.50, "%s_hundreds" % expid, "hundreds", False),
            ]

###

def post_jobs(administrator_URL, responses_rewards, duration,
              tc, exp, random_type = False):

    keys = []

    for responses, reward, admin_id, HIT_id, append in responses_rewards:
        questions = get_questions_for_responses(responses, append)

        jobs = None
        #if random_type:
        #    jobs = gen_jobs_random(questions, num_assignments_per_condition)
        #else:
        qr = zip(questions, get_num_participants(responses))
        print "Number of conditions", len(qr)
        jobs = gen_jobs_for_all_questions_long(qr)

        print len(jobs)

        r = add_jobs(administrator_URL, admin_id, jobs,
            'chicoritastranglelemon', timeout=duration + 60)
        print r
        print r.headers

        num_assignments_total = len(jobs)

        hit =  RandomStorm(responses,
                  num_assignments_total,
                  reward,
                  administrator_URL,
                  admin_id,
                  questions,
                  duration = duration,
                  append_ideas = append)

        key = start_trial(tc, hit, exp, HIT_id)
        keys.append(key)

    return keys

if __name__=='__main__':
    exp_location, schema, expid, tc, admin_url = initialize_from_cmd(script_name)

    with sl.Experiment(exp_location, schema, expid) as exp:

        keys = post_jobs(admin_url, get_response_rewards(expid), 60*60*18,
                  tc=tc, exp=exp, random_type = False)

        results = [attempt_finish_trial(tc, exp, key) for key in keys]
