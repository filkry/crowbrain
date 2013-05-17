import sqlite3, sys, datetime
from os.path import expanduser

class ExperimentSchema:
    def __init__(self, name, version, trial_rows, result_rows):
        self.table_name = "%s_%s" % (name, version)
        self.trial_ok = [v[0] for v in trial_rows]
        self.result_ok = [v[0] for v in result_rows]

        self.columns = [(u"experiment_id", u"text"),
                        (u"key", u"text"),
                        (u"trial_type", u"text"),
                        (u"datetime", u"text")] + trial_rows + result_rows

    def trial_contains_key(self, key):
        return key in self.trial_ok

    def test_equal(self, other):
        dcols = {}
        for col, t in self.columns:
            dcols[unicode(col)] = t

        for key, t in other:
            if not key in dcols:
                return False
            if unicode(dcols[key]) != t:
                return False
        return True

class Experiment:
    def __init__(self, fn, schema, expid):
        self.conn = sqlite3.connect(expanduser(fn))
        self.schema = schema
        self.expid = expid

        c = self.conn.cursor()

        existing_schema = []
        for row in c.execute("PRAGMA table_info(%s)" % (schema.table_name)):
            existing_schema.append((row[1], row[2]))

        if not schema.test_equal(existing_schema):
            res = None
            while res != 'Y' and res != 'N':
                res = raw_input("Warning: schema provided does not match existing table! Continue (Y/N)?")

            if res == 'N':
                sys.exit(0)

        c.execute("create table if not exists %s (%s)" %
                  (schema.table_name,
                   ', '.join(["%s %s" % (col[0], col[1]) for col in schema.columns] + ["PRIMARY KEY (experiment_id, key)"])))

        self.conn.commit()

    # enter and exit allow use with "with" syntax, auto cleanup
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.conn.close()

    def start_trial(self, key, trial_type, values):
        """ Start an experimental trial

        Arguments:
        trial_type -- generic string reference to type of trial
        values -- list of tuples to put in db
        key -- unique (with experiment_id) key by which to refer to the trial
        trial_func -- function to pass values to to start the trial

        Returns the key passed """

        for col, v in values:
            if not col in self.schema.trial_ok:
                print("Warning: starting trial with values not in schema")

        ts = values

        sql ="SELECT * FROM %s WHERE experiment_id=? and key=?" % (self.schema.table_name)
        for row in self.conn.execute(sql, (self.expid, key)):
            print("Trying to create trial with existing id+key, skipping")
            return

        # add auto-generated values
        ts = ts + [("experiment_id", self.expid),
                   ("key", key),
                   ("trial_type", trial_type),
                   ("datetime", str(datetime.datetime.now()))]

        sql = "INSERT INTO %s (%s) VALUES (%s)" % (self.schema.table_name,
                                                   ','.join([t[0] for t in ts]),
                                                   ','.join(['?'] * len(ts)))


        self.conn.execute(sql, [t[1] for t in ts])
        self.conn.commit()
        return key

    def finish_trial(self, key, values):
        """ Finish an experimental trial

        Arguments:
        key -- unique key (within experiment) for trial that has been started
        values -- list of tuples

        """

        for col, b in values:
            if not col in self.schema.result_ok:
                print("Warning: finishing trial with values not in schema")

        ts = values

        sql = '''update %s set %s where
                    experiment_id="%s" and
                    key="%s"''' % (self.schema.table_name,
                                   ','.join(["%s=?" % (t[0]) for t in ts]),
                                   self.expid,
                                   key)

        self.conn.execute(sql, [t[1] for t in ts])
        self.conn.commit()


if __name__ == '__main__':
    schema = ExperimentSchema("test", "v0", [("trialrow", "text"), ("trialrow2", "text")],
                                            [("resultrow1", "integer")])

    def mytrial(trialrow, trialrow2):
        print("Performing trial! %s, %s" % (trialrow, trialrow2))

    with Experiment("test.sqlite", schema) as experiment:

        experiment.start_trial("test_exp_id", "t1", [("trialrow", "filip"), ("trialrow2", "kry")], "key")
        mytrial("filip", "kry")
        experiment.finish_trial("test_exp_id", "key", [("resultrow1", 1000)])

        print("Done")
