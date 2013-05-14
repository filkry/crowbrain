#!/bin/bash

apt-get update
apt-get install -y python-pip python-dev build-essential sqlite3
pip install turkflow

ln -s -f /home/host/.boto /home/vagrant/.boto
