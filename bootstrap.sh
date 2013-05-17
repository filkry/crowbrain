#!/bin/bash

apt-get update
apt-get install -y python-pip python-dev build-essential sqlite3
pip install turkflow
pip install ipython

ln -s -f /home/host/.boto /home/vagrant/.boto

cd /home/host/enc_projects/crowbrain/ipython
ipython notebook --pylab inline
