import numpy as np
import xlsxwriter
import javalang
import sys

#print(sys.argv[1])
java_string = sys.argv[1]
f = open("error_and_line.txt", "r")
line_array = ["line number"]
error_array = ["error message"]
line_offset_forward = 5
line_offset_backward = 2
error_offset = 7

def get_excel(line_array,error_array,pos_array):
	total_array = [line_array,pos_array,error_array]
	workbook = xlsxwriter.Workbook('error_and_line.xlsx')
	worksheet = workbook.add_worksheet()

	row = 0

	for col, data in enumerate(total_array):
	    worksheet.write_column(row, col, data)

	workbook.close()


# get error info and matching line number of each error
def parse_error():
	for x in f:
		cur_x = str(x) 
		line_index = cur_x.find("java:")
		error_index = cur_x.find("error: ")
		if(line_index != -1 and error_index != -1):
			line_array.append(x[line_index + line_offset_forward : error_index - line_offset_backward])
		elif(error_index != -1):
			error_array.append(x[error_index + error_offset :].replace("\n",""))

parse_error()

#iinitialize line number+function name 
all_variable = []
all_variable.append([0,'N'])
temp = open(java_string,"r")
total_num_of_lines = 0
line_content = []
for y in temp:
	line_content.append(y)
	total_num_of_lines = total_num_of_lines + 1


file = open(java_string,"r")
string = file.read()
tree = javalang.parse.parse(string)

#record position(line number) of each token
tokens = list(javalang.tokenizer.tokenize(string))
tokens_list = {}
for i in range(len(tokens)):
	#print(tokens[i])
	if tokens[i].value in tokens_list:
		if("private" in line_content[int(tokens[i].position.line)-1] or 
			"public" in line_content[int(tokens[i].position.line)-1] or 
			"protected" in line_content[int(tokens[i].position.line)-1]):
			tokens_list[tokens[i].value] = int(tokens[i].position.line)
		#pass
	else:
		tokens_list[tokens[i].value] = int(tokens[i].position.line)


#helper function to parse body of each unit
def get_body(cur):
	for j in range(len(cur)):
		if(str(cur[j]).startswith("MethodDeclaration")):
			all_variable.append([tokens_list[cur[j].name],cur[j].name])
		if(hasattr(cur[j], 'body')):
			#print(type(cur[j].body))
			if((not str(cur[j].body).startswith("BlockStatement")) and 
				(not str(cur[j].body).startswith("EnumBody")) and 
				cur[j].body != None and len(cur[j].body) != 0):
				get_body(cur[j].body)


get_body(tree.types)

all_variable.append([total_num_of_lines,'N'])


# sort the method by line number from smallest to largest
def sort(input_array):
	output_array = [all_variable[0],all_variable[len(all_variable)-1]]
	#print(output_array)
	for i in range(1,len(all_variable)-1):
		#print(i,all_variable[i])
		cur_line_number = all_variable[i][0]
		for j in range(len(output_array)-1):
			#print(cur_line_number,output_array[j][0],output_array[j+1][0])
			if(cur_line_number > output_array[j][0] and cur_line_number <= output_array[j+1][0]):
				output_array[j+1:j+1] = [all_variable[i]]
			#print(output_array)
	return output_array

sort_variable = sort(all_variable)
original_variable = all_variable
all_variable = sort_variable

# helper function to find matching function name of that specific line number of error
def find_pos(num_of_line):
	for i in range(len(all_variable)-1):
		#print(all_variable[i][0],all_variable[i+1][0])
		if(num_of_line >= all_variable[i][0] and num_of_line < all_variable[i+1][0]):
			return all_variable[i][1]

#find matching function name of each error
pos_array = []
def find_pos_array():
	for i in range(len(line_array)):
		if(i == 0):
			pos_array.append("matched function name")
		else:
			pos_array.append(find_pos(int(line_array[i])))

find_pos_array()

#print(original_variable)
#print(sort_variable)
#print(pos_array)
#print(line_array)
get_excel(line_array,error_array,pos_array)


