train_file="train/"
validate_file="validate/"
test_file="test/"
EXT=java

for each_data_set in *; do
	for folder in "$each_data_set"/*; do
		for i in "$folder"/*; do
		    cur_file="$(basename "$i")"
		    java_file_name="${cur_file%%.*}"
		    
		    path_java="$folder"/"$java_file_name"/"$java_file_name".java
		    FILE="$folder"/"$java_file_name"/"$java_file_name".txt
			if grep -i -q "null" "$FILE"; then
  				echo "$java_file_name"
			fi

		done
	done
done