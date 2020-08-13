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
		    cur_file="$(basename "$i")"
		    java_file_name="${cur_file%%.java}"
		    java_file="$java_file_name".java
		    
		    FILE="$folder"/"$java_file_name"/error_and_line.txt
		    #rm -r "$folder"/"$java_file_name"/infer-out/
			if test -f "$FILE"; then
				#echo "begin parsing"
				#echo "$i"
				cp parse_all.py "$i"/.
				cd "$i"
				python3 parse_all.py "$java_file"
				rm parse_all.py

				if test -f error_and_line.xlsx; then
					echo "success"
				else
					echo "=========================="
					echo "$i"
					echo "=========================="
				fi
				cd -
			fi

		done
	done
#done