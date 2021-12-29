"""
12scenes dataset setup script taken from DSAC* (identical)
https://github.com/vislearn/dsacstar/blob/master/datasets/setup_12scenes.py

Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import zipfile

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

# name of the folder where we download the original 12scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '12scenes_source'

# download the original 12 scenes dataset for calibration, poses and images
mkdir(src_folder)
os.chdir(src_folder)

for ds in ['apt1', 'apt2', 'office1', 'office2']:

	print("=== Downloading 12scenes Data:", ds, "===============================")

	os.system('wget http://graphics.stanford.edu/projects/reloc/data/' + ds + '.zip')

	# unpack and delete zip file
	f = zipfile.PyZipFile(ds + '.zip')
	f.extractall()

	os.system('rm ' + ds + '.zip')

	scenes = os.listdir(ds)

	for scene in scenes:

		data_folder = ds + '/' + scene + '/data/'

		if not os.path.isdir(data_folder):
			# skip README files
			continue

		print("Linking files for 12scenes_" + ds + "_" + scene + "...")

		target_folder = '../12scenes_' + ds + '_' + scene + '/'

		# create subfolders for training and test
		mkdir(target_folder + 'test/rgb/')
		mkdir(target_folder + 'test/poses/')
		mkdir(target_folder + 'test/calibration/')

		mkdir(target_folder + 'train/rgb/')
		mkdir(target_folder + 'train/poses/')
		mkdir(target_folder + 'train/calibration/')		

		# read the train / test split - the first sequence is used for testing, everything else for training
		with open(ds + '/' + scene + '/split.txt', 'r') as f:
			split = f.readlines()
		split = int(split[0].split()[1][8:-1])

		# read the calibration parameters, we use only the focallength
		with open(ds + '/' + scene + '/info.txt', 'r') as f:
			focallength = f.readlines()
		focallength = focallength[7].split()
		focallength = (float(focallength[2]) + float(focallength[7])) / 2

		files = os.listdir(data_folder)

		images = [f for f in files if f.endswith('color.jpg')]
		images.sort()

		poses = [f for f in files if f.endswith('pose.txt')]
		poses.sort()

		def link_frame(i, variant):
			""" Links calibration, pose and image of frame i in either test or training. """

			# some image have invalid pose files, skip those
			valid = True
			with open(ds + '/' + scene + '/data/' + poses[i], 'r') as f:
				pose = f.readlines()
				for line in pose:
					if 'INF' in line:
						valid = False

			if not valid:
				print("Skipping frame", i, "("+variant+") - Corrupt pose.")
			else:
				# link pose and image
				os.system('ln -s ../../../' + src_folder + '/' + data_folder + '/' + images[i] + ' ' + target_folder + variant + '/rgb/')
				os.system('ln -s ../../../' + src_folder + '/' + data_folder + '/' + poses[i] + ' ' + target_folder + variant + '/poses/')

				# create a calibration file
				with open(target_folder + variant  + '/calibration/frame-%s.calibration.txt' % str(i).zfill(6), 'w') as f:
					f.write(str(focallength))

		# frames up to split are test images
		for i in range(split):
			link_frame(i, 'test')

		# all remaining frames are training images
		for i in range(split, len(images)):
			link_frame(i, 'train')
