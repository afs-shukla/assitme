import json
with open("info_solution.json",encoding='utf-8') as read_file:
        data = json.load(read_file)

a=data['items']

title=[]

for d in a:
	title.append((d['title']))

file=open("entity_data.txt","x")
for i in title:
	file.write(i+"\n")
file.close()


