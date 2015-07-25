from pathlib import Path
import subprocess
import os
import time
import datetime

executable = 'RayTraversal_x64_Release.exe'
kernel = '--kernel=kepler_dynamic_fetch'

class result:
	def __init__(self):
		self.name = ''
		self.sah = 0
		self.time = 0
		self.memory = 0
		self.mrays = 0
		
def makepath(tuple):
	path = ''
	for part in tuple:
		path += (part)
		path += '/'
	return path

def GenerateReports(id, experimentName):
	for objFilePath in Path('scenes').glob('**/*.obj'):
		objFolder = objFilePath.parts[0:len(objFilePath.parts)-1]
		
		# Read camera file
		camFilePath = list(Path(makepath(objFolder)).glob('*.cam'))[0]
		with open(str(camFilePath)) as f:
			content = f.readlines()
		if (len(content) == 0):
			continue
		cameras = ''

		
		# Assemble command
		command = [executable]
		command.append('benchmark')
		command.append('--mesh=' + str(objFilePath) + '')
		for line in content:	
			command.append('--camera=' + line.rstrip('\r\n') + '')
		command.append(kernel)
		
		# Run and save log file

		process = subprocess.Popen(command, stdout=subprocess.PIPE)
		out, err = process.communicate()
		reportFileName = objFilePath.name.replace('.obj', '.txt')
		reportDirectory = 'experiments/' + id + '/' + experimentName + '/reports/' + makepath(objFilePath.parts[1:len(objFilePath.parts)-1])
		reportPath = reportDirectory + reportFileName
		if not os.path.exists(reportDirectory):
			os.makedirs(reportDirectory)
		with open(reportPath, 'w+') as f:
			f.write(out.decode('ascii'))
			
def ParseReports(id, experimentName):
	entries = []
	for reportFilePath in Path('experiments/' + id + '/' + experimentName + '/reports').glob('**/*.txt'):
		with open(str(reportFilePath)) as f:
			content = f.readlines()
			
		entry = result()
		
		# Get SAH
		for line in content:
			if '\tSAH: ' in line:
				entry.sah = line.rstrip().split('\tSAH: ')[1]
				
		# Get execution time
		time = 0
		for line in content:
			if '\tBuild time: ' in line:
				time += float(line.rstrip().split('\tBuild time: ')[1].split(' ms')[0])
			if '\tOptimize time: ' in line:
				time += float(line.rstrip().split('\tOptimize time: ')[1].split(' ms')[0])
			if '\tCollapse time: ' in line:
				time += float(line.rstrip().split('\tCollapse time: ')[1].split(' ms')[0])
		entry.time = str(time)
				
		# Get used memory
		memory = 0.0
		for line in content:
			if '\tGlobal memory used: ' in line:
				memory = float(line.rstrip().split('\tGlobal memory used: ')[1].split(' B')[0])
		entry.memory = str(memory / (1024 * 1024))
		
		# Get MRays/s
		for line in content:
			if 'Mrays/s = ' in line:
				entry.mrays = line.rstrip().split('Mrays/s = ')[1]
		
		# Get name
		entry.name = reportFilePath.name.split(reportFilePath.suffix)[0]
		
		entries.append(entry)
		
	# Write CSV report
	with open('experiments/' + id + '/' + experimentName + '/reports/report.csv', 'w+') as f:
		# Header
		for entry in entries:
			for index in range(0, 4):
				f.write(entry.name)
				f.write(', ')
		f.write('\n')
		# Header 2
		for entry in entries:
			f.write('MRays/s')
			f.write(', ')
			f.write('SAH')
			f.write(', ')
			f.write('Time (ms)')
			f.write(', ')
			f.write('Memory (MB)')
			f.write(', ')
		f.write('\n')
		
		# Values
		for entry in entries:
			f.write(entry.mrays)
			f.write(', ')
			f.write(entry.sah)
			f.write(', ')
			f.write(entry.time)
			f.write(', ')
			f.write(entry.memory)
			f.write(', ')		
		f.write('\n')
		
def JoinResults(id):
	results = []
	addedHeader = False
	for reportFilePath in Path('experiments/' + id).glob('**/report.csv'):
		experimentName = str(reportFilePath).split('\\')[2]
		with open(str(reportFilePath)) as f:
			content = f.readlines()
		
			if not addedHeader:
				results.append(' , ' + content[0])
				results.append(' , ' + content[1])
				addedHeader = True
		
			results.append(experimentName + ', ' + content[2])
	
	with open('experiments/' + id + '/consolidated.csv', 'w+') as f:
		for line in results:
			f.write(line)
		
# Run script
id = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

# LBVH
name = 'LBVH'
with open('bvhrt.cfg', 'w+') as f:
	f.write('lbvh64 collapse')
GenerateReports(id, name)
ParseReports(id, name)

# TRBVH
name = 'TRBVH'
with open('bvhrt.cfg', 'w+') as f:
	f.write('lbvh64 trbvh treeletSize=7 iterations=3 collapse')
GenerateReports(id, name)
ParseReports(id, name)

# ATRBVH
'''
for i in range(4, 33):
	for j in range(4, 6):
		name = 'ATRBVH-' + str(i) + '_' + str(j)
		with open('bvhrt.cfg', 'w+') as f:
			f.write('lbvh64 atrbvh treeletSize=' + str(i) + ' iterations=' + str(j) + ' collapse')
		GenerateReports(id, name)
		ParseReports(id, name)
'''
name = 'ATRBVH-9_2'
with open('bvhrt.cfg', 'w+') as f:
	f.write('lbvh64 atrbvh treeletSize=9 iterations=2 collapse')
GenerateReports(id, name)
ParseReports(id, name)

name = 'ATRBVH-7_3'
with open('bvhrt.cfg', 'w+') as f:
	f.write('lbvh64 atrbvh treeletSize=7 iterations=3 collapse')
GenerateReports(id, name)
ParseReports(id, name)

JoinResults(id)