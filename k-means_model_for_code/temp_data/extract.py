import xlrd
import pandas as pd
import numpy as np
import sys

range_offset = 2;
def get_error_line():
	#workbook = xlrd.open_workbook("error_and_line.xlsx")
	#sheet = workbook.sheet_by_index(0)

	#for rowx in range(sheet.nrows):
	#    values = sheet.row_values(rowx)
	#    print(values)


	xlsx = pd.ExcelFile("error_and_line.xlsx")
	sheetX = xlsx.parse(0)
	var1 = sheetX['line number']

	#print(var1[0])
	#print(len(var1))
	return var1[0]

def get_error_type():
	xlsx = pd.ExcelFile("error_and_line.xlsx")
	sheetX = xlsx.parse(0)
	var1 = sheetX['error message']

	#print(var1[0])
	return var1[0]

def get_context_arr(line_number):
	java_file = sys.argv[1] + ".java"
	context_range = []
	index_of_error_context = line_number-1
	with open(java_file , 'r') as f:
	    lines = f.readlines()
	    context_range = lines[index_of_error_context-range_offset:index_of_error_context+range_offset+1]
	    f.close()
	#for i in range(len(context_range)):
	#	print(context_range[i])
	return context_range

def wrap_output_string(context_arr,error_type):
	output_string = "public void " + error_type + "(){\n"
	for i in range(len(context_arr)):
		output_string += context_arr[i]
	output_string += "\n}"
	return output_string

line_number = get_error_line()
error_type = get_error_type()
#print(line_number,error_type)
context_arr = get_context_arr(line_number)
extracted_string = wrap_output_string(context_arr,error_type)
#print(extracted_string)
new_file = open("Output.java", "w")
new_file.write(extracted_string)
new_file.close()



