#!/bin/bash

apt-get update
apt-get install -y python-pip python-dev python-requests build-essential sqlite3 libfreetype6-dev libpng-dev python-scipy python-matplotlib python-tornado python-zmq python-nltk pandoc python-pygments python-sphinx pyside-tools r-base graphviz graphviz-dev pkg-config
pip install --upgrade ipython
pip install turkflow numpy pyyaml pandas pygraphviz

ln -s -f /home/host/.boto /home/vagrant/.boto

mkdir -p /home/vagrant/.ipython
chown vagrant /home/vagrant/.ipython

mkdir -p /home/vagrant/scratch
chmod 777 /home/vagrant/scratch

if [[ ! -e /home/vagrant/.ipython/profile_server ]]; then
	ln -s -f /vagrant/ipython_profile_server /home/vagrant/.ipython/profile_server
fi
