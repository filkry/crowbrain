# Contains a bunch of common functionality that doesn't change between tests

import scilog as sl
from turkflow.turkflow import *
from jinja2 import *
import json, re, argparse

class MyUndefined(Undefined):
    def __getattr__(self, name):
        return ''

def guess_autoescape(template_name):
    if template_name is None or '.' not in template_name:
        return False
    ext = template_name.rsplit('.', 1)[1]
    return ext in ('html', 'htm', 'xml')

env = Environment(autoescape=guess_autoescape,
                  loader=PackageLoader('common', 'templates'),
                  undefined=MyUndefined,
                  extensions=['jinja2.ext.autoescape'])

# Structs
# Results processing

def timeout():
    print("Timed out waiting for hits, run again later")
    sys.exit(0);

# TODO: requires hit to have already been processed by create_hit, big oversight
def get_trial_vals(hit, schema):
    d = hit.__dict__
    ret = [(col, json.dumps(d[col])) for col in d if schema.trial_contains_key(col)]
    return ret

# TODO: this is ugly and filled with redundant doing
def start_trial(tc, hit, exp, trial_type, reset_phase = 0):
    key = tc.createHIT(hit, reset_counter=reset_phase)
    exp.start_trial(key, trial_type, get_trial_vals(hit, exp.schema) + [('hit_id', key)])
    return key

def attempt_finish_trial(tc, exp, key):
    results, times = tc.waitForHIT(key, timeout=0)
    if not results:
        timeout()
    exp.finish_trial(key, [("responses_json", json.dumps(results)),
                           ("accept_times", json.dumps([t['acceptTime'] for t in times])),
                           ("submit_times", json.dumps([t['submitTime'] for t in times]))])
    return results

