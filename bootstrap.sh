#!/bin/bash

apt-get update
apt-get install -y python-pip python-dev build-essential sqlite3 libfreetype6-dev libpng-dev
pip install turkflow pyzmq tornado ipython matplotlib

ln -s -f /home/host/.boto /home/vagrant/.boto

mkdir -p /home/vagrant/.ipython
ln -s -f /vagrant/ipython_profile_server /home/vagrant/.ipython/profile_server
