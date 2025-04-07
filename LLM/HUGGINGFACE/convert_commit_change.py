import re

with open('temp_commit_log.txt', 'r') as file:
    log = file.read()

versions = re.split(r'v\d+\.\d+\.\d+', log)
for idx, version in enumerate(versions, 1):
    print(f"## [v{idx}]\n{version}\n")

