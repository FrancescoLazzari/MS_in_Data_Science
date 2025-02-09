#!/bin/bash

# /!\ ATTENTION /!\

# Into the Q2 we used the 'gnuplot' command which is not presente natively in macOS
# So if you are using a Mac you need to install the gnuplot manually

# We can do this with a packages manager for macOS like Homebrew
# To install Homebrew we can pasting and executing the following command into the terminal

# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# This command can also been found on this link --> https://brew.sh/

# After installing Homebrew we can install gawk by pasting and executing the following command in the terminal
# brew install gnuplot

echo 
echo "ADM-HW5 Winter Semester 2023"
echo 
echo "# CLQ - Group 24"
echo 
echo "## Answer the following questions: "
echo 
echo "-------------------------------------------------------------------------------------------------------"
echo " Q1) Is there any node that acts as an important 'connector' between the different parts of the graph? "
echo "-------------------------------------------------------------------------------------------------------"
echo 
echo "- Node Degree Centrality: "
echo 
# File name
graph="citation_subgraph.edgelist"

# Calculate and print the Degree-Centrality table
calculate_degree_centrality(){

    # Sub-function to print a separation line automatically
    print_line(){
        printf '=%.0s' {1..36}
        printf '\n'
    }
    # 'printf'      is a command to print in a formatted output
    # '='           is the character to be printed
    # '%.0s'        specifies that a string (s) is printed with length 0 (0)
    # '.'           indicates that we do not want to print any character from the input string (since there is none)
    # '{1..36}'     repeats the print command 36 times
 
    # Calculate the total number of nodes in the Graph
    total_nodes=$(cat "$graph" | awk '{print $1; print $2}' | sort | uniq | wc -l)
    # 'cat'     reads the file
    # 'awk'     extracts and prints the nodes (columns 1 and 2 of each row)
    # 'sort'    sorts the nodes
    # 'uniq'    removes duplicate nodes
    # 'wc -l'   counts the remaining rows (unique nodes)

    print_line
    # 'printf' formats and prints the table header
    printf "  %-10s  |  %-8s \n" "Node ID" "Degree Centrality"
    # '%-10s' and '%-8s' align a strings to the left (-) in a field of 10 and 8 characters respectively.
    print_line

    # Calculate the Degree-Centrality for each node
    cat "$graph" | awk '{print $1; print $2}' | sort | uniq -c | \
    awk -v total="$total_nodes" '{centrality = $1 / (total - 1); printf "  %-10s  |  %-15f \n", $2, centrality}' | \
    sort -nr -k3 | head -n 6
    # 'cat'         reads the file and outputs nodes (columns 1 and 2 of each row)
    # 'awk'         extracts and prints the nodes
    # 'sort'        sorts the nodes so that 'uniq -c' can count the occurrences 
    # 'uniq -c'     counts occurrences of each node (degree)

    # 'awk -v total="$total_nodes"'    passes the total number of nodes to the 'awk' function
    # 'awk'             calculates the degree centrality and formats the output
    #                   it uses the first column ($1) since it is the degree count for each node
    #                   the second column ($2) is the specific node for which we have calculated the centrality 
    # 'sort -nr -k3'    sorts the results by degree-centrality (column 3) in descending order
    # 'head -n 6'       selects the top 6 nodes 


    print_line
}

# Call the 'calculate_degree_centrality' function to compute and display the results
calculate_degree_centrality

echo 
echo "------------------------------------------------------------------"
echo " Q2) How does the degree of citation vary among the graph nodes? "
echo "------------------------------------------------------------------"
echo 
echo "- In/Out Degree variation:"
echo 

# Function to calculate and print the frequencies of in-degrees and out-degrees
calculate_degree_distribution(){
    local column=$1      # Assigns the first parameter of the function to the variable 'column' (column to analyze)
    local title=$2       # Assigns the second parameter of the function to the variable 'title' (title of the distribution)
    local file_name=$3   # Assigns the third parameter of the function to the variable 'file_name' (name of the output file)

    # Sub-function to print a separation line automatically
    print_line(){
        printf '=%.0s' {1..27}
        printf '\n'
    }
    # 'printf'      is a command to print in a formatted output
    # '='           is the character to be printed
    # '%.0s'        specifies that a string (s) is printed with length 0 (0)
    # '.'           indicates that we do not want to print any character from the input string (since there is none)
    # '{1..36}'     repeats the print command 36 times

    # Extraction and calculation of frequencies
    awk -v col="$column" '{print $col}' "$graph" | sort | uniq -c | \
    awk '{print $1}' | sort | uniq -c | sort -k2n > "$file_name"
    # 'awk -v col="$column"'     sets an internal awk variable ('col') with the value of "$column"
    #                            which indicates the column to analyze (1 for out-degree, 2 for in-degree)
    # '{print $col}'             extracts the relevant nodes
    # 'sort'                     sort the nodes so that each row/edge of the same node is in consecutive rows
    # 'uniq -c'                  counts (-c) how many times each individual node (uniq) appears
    #                            returns the in/out degree count for each node

    # 'awk '{print $1}'          extracts only the degree number for each node
    # 'sort | uniq -c'           sort the degree number so that nodes with the same degree are in consecutive rows 
    #                            it returns a pair of (frequency of nodes with those degrees; number of degrees)
    # 'sort -k2n'                sort the result based on the degree number (column 2) in ascending order
    # "> $file_name"             saves the final output in a file named as specified by the "$file_name" variable
    #                            this file will contain a pair of frequency-degree on each line
    #                            this file is necessary to pass the data to the 'gnuplot' function

    # Print the degree frequency table
    print_line  
    # Prints the table header calling the first column title with the same name 
    # passed by the second argument of the function
    printf "  %-10s |  %-8s \n" "$title" "Frequency"  
    print_line 
    awk '{printf "  %-10s |  %-8s \n", $2, $1}' "$file_name"  
    # 'awk' reads the file with the frequencies and prints the formatted table
    # '%-10s' and '%-8s' allign a strings to the left (-) in a field of 10 and 8 characters respectively

    print_line

}

# Setting file names for in/out degree frequencies files
in_degree_file="in_degree_distribution.txt"
out_degree_file="out_degree_distribution.txt"

# Calculate and print frequency tables
calculate_degree_distribution 2 "In-Degree" "$in_degree_file"    # Calculate the distribution for the In-Degrees
echo
calculate_degree_distribution 1 "Out-Degree" "$out_degree_file"  # Calculate the distribution for the Out-Degrees
echo 

# Print the histogram using the 'gnuplot' command
gnuplot -e "set terminal png size 1024,768; set output 'out_degree_histogram.png'; \
set title 'In-Degree Frequency Distribution'; \
set xlabel 'Number of Degree in a node'; set ylabel 'Frequency'; \
set xrange [0:]; set yrange [0:]; \
set grid ytics lt 0 lw 1 lc rgb '#DDDDDD'; \
set style histogram; set style fill solid 1.00 border -1; set boxwidth 0.9; \
set border 3; set tics out nomirror; \
unset key; \
plot '$in_degree_file' using 2:1:xtic(2) with boxes lc rgb '#1E90FF'"

gnuplot -e "set terminal png size 1024,768; set output 'out_degree_histogram.png'; \
set title 'Out-Degree Frequency Distribution'; \
set xlabel 'Number of Degree in a node'; set ylabel 'Frequency'; \
set xrange [0:]; set yrange [0:]; \
set grid ytics lt 0 lw 1 lc rgb '#DDDDDD'; \
set style histogram; set style fill solid 1.00 border -1; set boxwidth 0.9; \
set border 3; set tics out nomirror; \
unset key; \
plot '$out_degree_file' using 2:1:xtic(2) with boxes lc rgb '#1E90FF'"

# Remove the distribution files
rm "$in_degree_file" "$out_degree_file"

echo "-------------------------------------------------------------------"
echo " Q3) What is the average length of the shortest path among nodes? "
echo "-------------------------------------------------------------------"
echo 

# Use Python and the Networkx library to calculate the average length of the shortest path among nodes
# the command '<<EOF' allows to write a Python code on multiple rows in a Bash file 
python3 <<EOF
import networkx as nx
citation_subgraph = nx.read_edgelist('citation_subgraph.edgelist', create_using=nx.DiGraph()) 
print(f'- The average shortest path length among nodes is: {round(nx.average_shortest_path_length(citation_subgraph),2)}') 
EOF

echo 

