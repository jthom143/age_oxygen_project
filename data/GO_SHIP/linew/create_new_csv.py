## Script to load in csv files Darryn send me on May 12, 2017 and create new csv files with both oxygen, age, and cruise dates 

import csv 
import glob

allFiles = glob.glob("*.dat")


for file_ in allFiles:
    filename = file_
    output = file_+'.csv'

    with open(filename) as csvin:
        readfile = csv.reader(csvin, delimiter=' ')
        first_line = next(readfile)

        with open(output, 'w') as csvout:
            writefile = csv.writer(csvout, delimiter=' ', lineterminator='\n')

            for row in readfile:
                if row[0] == "STN":
                    row.extend(['DATE'])
                    writefile.writerow(row)
                else:
                    row.extend(first_line)
                    writefile.writerow(row)
