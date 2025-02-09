#!/bin/bash

echo -e "\n"
echo "ADM-HW3 Winter Semester 2023"
echo -e "\n"
echo "# CLQ - Group 15"
echo -e "\n"
echo "## Merge all the course_i.tsv file into a file named merged_course.tsv"

# Create a new .tsv file named 'merged_course' with specific columns

echo -e "courseName\tuniversityName\tfacultyName\tisItFullTime\tdescription\tstartDate\tfees\tmodality\tduration\tcity\tcountry\tadminsitration\turl" > merged_courses.tsv

for i in {1..400}; do
    folder="Page$i"
    # Move into the folder or if the folder doesnt exist terminates the script
    cd "$folder" || exit

    for j in {0..14}; do
        file="course_$j.tsv"
        # All the couse_i file have only the row with the value without the headers
        # If the sile doesnt exist terminates the script
        cat "$file" >> ../merged_courses.tsv || exit
        # This command is necessary to prevent new lines from being added all to the first line
        echo -e "\n" >> ../merged_courses.tsv  

    done
    # Move back into the previous folder
    cd ..
done

# grep -v -E '^[[:space:]]*$' merged_courses.tsv --> It removes lines that consist only of whitespace or are completely empty
# Then we overwrite the merged_couses file passing through a temporary file
grep -v -E '^[[:space:]]*$' merged_courses.tsv > merged_courses_temp.tsv && mv merged_courses_temp.tsv merged_courses.tsv

# If we get to this point all the files have been merged correctly
# If a folder is missing the exit command stops execution making it impossible to get to this line
echo "Done ✅"
#---------------------------------------------------------------------

echo -e "\n"
echo "## Answer the following questions:"
echo -e "\n"
echo "Q1) Which city and wich county offers the most Master's Degree? "

# The 'awk' command allows to execute a action specified between braces to all the lines of a file that respect a certain pattern ---> example:  awk 'pattern { action }' file
# -F'\t'  --> Specifies the field separator as a tab character
# NR>1    --> Ensures that the actions inside the block are only applied to lines with a record number greater than 1 (it skipps the header) 
# gsub(/ /, "_", $11)  -->  Replaces spaces with underscores in the 11th column 
#                           --> This step is essential to ensure that occurrences are sorted correctly since the presence of empty spaces creates problems in sorting
# count[$11]++         -->  Increments the count for each value in the 11th column
# END {for (country in count) print country, count[country]} --> Iterates over the array of counts and prints each country along with its count
# sort -k2,2nr  --> Sorts based on the second column (count) in reverse numerical order
# head -n1      --> Output only the first line
# read -r country count1 <<< --> Assigns the output to the two variables  country and count1

read -r country count1 <<< "$(awk -F'\t' 'NR>1 { gsub(/ /, "_", $11); count[$11]++ } END {for (country in count) print country, count[country]}' merged_courses.tsv | sort -k2,2nr | head -n1)"

# remove the first _ from the left
country="${country#_}"
# remove the first _ from the right
country="${country%_}"
# replace all the remaning _ with a space
country="${country//_/ }"

# Print the result
echo "- The country that offer the most Master's Degree is $country with $count1 courses"

# Does the same but on the 10th column
read -r city count2 <<< "$(awk -F'\t' 'NR>1 { gsub(/ /, "_", $10); count[$10]++ } END {for (city in count) print city, count[city]}' merged_courses.tsv | sort -k2,2nr | head -n1)"

city="${city#_}"
city="${city%_}"
city="${city//_/ }"

echo "- The city that offer the most Master's Degree is $city with $count2 couses"

#---------------------------------------------------------------------
echo -e "\n"
echo "Q2) How many colleges offer Part-Time education?"

# The 'awk' command allows to execute a action specified between braces to all the lines of a file that respect a certain pattern ---> example:  awk 'pattern { action }' file
# -F'\t'  --> Specifies the field separator as a tab character
# NR>1    --> Ensures that the actions inside the block are only applied to lines with a record number greater than 1 (it skipps the header) 
# $4 ~ /Part time/  --> Check if the value in the 4th column contains the string "Part time"
# !arr[$2]++        --> Counting how many unique values are in the 2nd column
#                       -->  arr[$2]  --> Accesses the value stored in the array at the 2nd column.
#                       -->  ! ... ++ -->  Check if this value hasn't been encountered before and then increments the value in the array 
# college_part_time=$(...)  -->  Store the result of the command into the college_part_time variable

college_part_time=$(awk -F'\t' 'NR>1 && $4 ~ /Part time/ && !arr[$2]++ {count++} END {print count}' merged_courses.tsv)

echo "- The number of different colleges that offer a part time education is: $college_part_time"

#---------------------------------------------------------------------
echo -e "\n"
echo "Q3) What is the percentage of courses in Engineering?"

# The 'awk' command allows to execute a action specified between braces to all the lines of a file that respect a certain pattern ---> example:  awk 'pattern { action }' file
# -F'\t'  --> Specifies the field separator as a tab character
# NR>1    --> Ensures that the actions inside the block are only applied to lines with a record number greater than 1 (it skipps the header) 
# $1 ~ /Engineering/  --> Check if the value in the 1st column contains the string "Engineering"
# { count++ }         --> Count the number of rows matching the pattern
# END { print count } --> Print the count
# favorable_cases=$(...)  -->  Store the result of the command into the favorable_cases variable

favorable_cases=$(awk -F'\t' 'NR>1 && $1 ~ /Engineering/ { count++ } END { print count }' merged_courses.tsv)

# wc -l  --> Counts the number of lines in a file
total_cases=$(wc -l < merged_courses.tsv)

# scale=2  --> Set values to two decimal places
# | bc     --> Allows to make mathematical calculations
percentage=$(echo "scale=2; ($favorable_cases * 100) / $total_cases" | bc)

echo "- The percentage of courses in Engineering is: $percentage%"
echo -e "\n"

#rm merged_courses.tsv
