import re

# Open the file containing the commit log
with open('temp_commit_log.txt', 'r') as file:
    commit_log = file.readlines()

# Filter out commit messages with the word "merge" (case-insensitive)
filtered_log = [line for line in commit_log if not re.search(r'\bmerge\b', line, re.IGNORECASE)]

# Save the filtered commits to a new file or print them
with open('filtered_commit_log.txt', 'w') as output_file:
    output_file.writelines(filtered_log)

print("Filtered commit log saved to 'filtered_commit_log.txt'.")

