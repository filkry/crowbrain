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

###
# HIT Types
###

class RandomStorm(TurkHITType):
    def __init__(self, num_responses):
        self.num_responses = num_responses

        template = env.get_template("consent_plaintext")
        description = template.render({'num_responses': num_responses})

        TurkHITType.__init__(self,
            "Brainstorm %i ideas for a classic brainstorming problem." % (num_responses),
            string.split('research, brainstorming'),
            description = description,
            duration = (10 + num_responses) * 60,
            max_assignments = 300,
            annotation = 'brainstorm',
            reward = 0.50 if num_responses <= 15 else 1.00,
            env = env)

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

def initialize_from_cmd(script_identifier = "noscript"):
    parser = argparse.ArgumentParser(description='A test script for crowex')
    parser.add_argument('-usa', action='store_true',
            help='Restrict HITs to residents of the USA')
    parser.add_argument('-l', '--live', action='store_true',
            help='Send HITs to live MTurk site instead of sandbox')
    parser.add_argument('id', metavar='ID', nargs=1,
            help='An ID for the experiment')
    parser.add_argument('-db', '--database', action='store', default='~/scratch/crowbrain',
            help='The location to store database of outstanding hits. Program will hang if folder does not exist!')
    parser.add_argument('-r', '--reset', metavar='RESET_PHASE', action='store', type=float,
            help='Reset the experiment, removing all prior hits. If using, pass a stupid reset phase thingy')

    args = parser.parse_args()

    expid = args.id[0]

    db_location = "%s/%s.%s.%s.jobs" % (args.database, script_identifier, args.id[0], "sandbox" if not args.live else "live")
    exp_location = "%s/%s.%s.%s.sqlite" % (args.database, script_identifier, args.id[0], "sandbox" if not args.live else "live")

    print("Using db file %s" % (db_location))
    print("Using experiment file %s" % (exp_location))

    tc = TurkConnection(args.id[0], db_location, not args.live, args.reset,
            extra_files=['templates/Mop.svg.png', 'templates/bootstrap.css', 'templates/main.css'],
            us_only=args.usa)

    # Initialize
    schema = sl.ExperimentSchema(
            "crowbrain",
            "v0",
            [('hit_id', 'text'),
             ('title', 'text'),
             ('description', 'text'),
             ('keywords', 'text'),
             ('duration', 'integer'),
             ('assignments', 'integer'),
             ('reward', 'real'),
             ('template_name', 'text'),
             ('html', 'text')],
            [('responses_json', 'text'),
             ('accept_times', 'text'),
             ('submit_times', 'text')])

    return (exp_location, schema, expid, tc)


