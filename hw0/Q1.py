import csv
import sys

columnIndex = int(sys.argv[1]) + 1 # Because #0 col is ''
filename = sys.argv[2]

with open(filename, 'rb') as file:
	# Read input file to get an array of the specific column
	reader = csv.reader(file, delimiter=' ')
	elementList = []
	for row in reader:
		element = float(row[columnIndex])
		elementList.append(element)

	# Sort the element list
	elementList.sort()

	# Generate output with comma-seperated elements
	OUTPUT_FILENAME = 'ans1.txt'
	with open(OUTPUT_FILENAME, 'wb') as outputFile:
		writer = csv.writer(outputFile, delimiter = ',')
		writer.writerow(elementList)
