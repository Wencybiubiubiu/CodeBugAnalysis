train_file="train/"
validate_file="validate/"
test_file="test/"
EXT=java

#for each_data_set in *; do
	#echo "$each_data_set"
	#for folder in "$each_data_set"/*; do
	for folder in ./*; do
		echo "$folder"
		for i in "$folder"/*; do
			echo "$i"
		    cur_file="$(basename "$i")"
		    java_file_name="${cur_file%%.*}"
		    
		    path_java="$folder"/"$java_file_name"/"$java_file_name".java
		    FILE="$folder"/"$java_file_name"/"$java_file_name".txt
		    #rm -r "$folder"/"$java_file_name"/infer-out/
			if test -f "$FILE"; then
				echo "begin catch:"
			    error_message="$(cat "$FILE" | grep -o 'error:.*')"
			    echo "$error_message" > "$folder"/"$java_file_name"/error_and_line.txt
			    line_info="$(cat "$FILE" | grep -o 'java:.*')"
			    echo "$line_info" >> "$folder"/"$java_file_name"/error_and_line.txt
			fi

		done
	done
#done