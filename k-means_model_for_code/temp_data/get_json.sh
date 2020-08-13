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
		    #rm -r "$folder"/"$java_file_name"/infer-out/
			if test -f "$FILE"; then
			    #echo "$java_file_name"
			    #echo "$i"
			    #gtac "$FILE" |egrep -m 2 . | tail -1 | awk '{print $1;}'
			    #gtac "$FILE" |egrep -m 2 . | head -2 | awk '{print $1;}'
			    error_number="$(gtac "$FILE" |egrep -m 2 . | tail -1 | awk '{print $1;}')"
			    error_file_line_count="$(cat "$FILE" | wc -l | xargs)"
			    java_file_line_count="$(cat "$path_java" | wc -l | xargs)"

			    cannot_find_symbol="$(grep -o "cannot find symbol" "$FILE" | wc -l| xargs)"
			    package_not_exist="$(grep -o "error: package" "$FILE" | wc -l| xargs)"
			    method_not_override="$(grep -o "error: method does not override" "$FILE" | wc -l| xargs)"
			    static_import="$(grep -o "error: static import only from classes and interfaces" "$FILE" | wc -l| xargs)"

			    if test -f "$folder"/"$java_file_name"/error_info.json; then
			    	rm "$folder"/"$java_file_name"/error_info.json
			    fi
			    echo "{\"static_import\":${static_import}\"cannot_find_symbol\":${cannot_find_symbol},\"package_not_exist\":${package_not_exist},\"method_not_override\":${method_not_override},\"error_number\":${error_number},\"error_file_line_count\":${error_file_line_count},\"java_file_line_count\":${java_file_line_count}}" > "$folder"/"$java_file_name"/"$java_file_name".json
			    
			    zero=0
			    if [ "$static_import" != "$zero" ] || [ "$method_not_override" != "$zero" ] || [ "$cannot_find_symbol" != "$zero" ] || [ "$package_not_exist" != "$zero" ];then
				    difference="$(($error_number - "$static_import" - $method_not_override - $cannot_find_symbol - $package_not_exist))"
			
				    if [ "$difference" != "$zero" ]; then
				    	echo "$java_file_name"
				    	echo "$difference"
				    	cat "$folder"/"$java_file_name"/"$java_file_name".json
				    fi
				fi
			fi

		done
	done
done