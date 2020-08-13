#!/bin/bash
train_file="train/"
validate_file="validate/"
test_file="test/"
EXT=java

for each_data_set in *; do
	for folder in "$each_data_set"/*; do
		for i in "$folder"/*; do
		    cur_file="$(basename "$i")"
		    java_file_name="${cur_file%%.*}"
		    mkdir "$folder"/"$java_file_name"
		    mv "$i" "$folder"/"$java_file_name"/.
		    cd "$folder"/"$java_file_name"
		    infer run -- javac "$cur_file" &> "$java_file_name".txt
		    cd -
		    #infer run -- javac "$cur_file" &> "$folder"/"$java_file_name"/"$java_file_name".txt
		done
	done
done