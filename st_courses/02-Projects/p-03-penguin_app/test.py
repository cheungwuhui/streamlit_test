import json
import requests

seed = 10
max_length = 100
word_input =  "何承博"
dataList=dict([('seed',seed),('max_length',max_length),("input",word_input)])
dataList_json=json.dumps(dataList)
url="http://180.184.50.70:21881/gpt2"

res = requests.post(url=url, data=dataList_json)
output = res.text
output = json.loads(output)
print(output['output'])
# print(output)