import requests
import json

# import test data
test_df = df.sample(5) # data is the pandas dataframe of the original data
label_df = test_df.pop('DEATH_EVENT')

test_sample = json.dumps({'data': test_df.to_dict(orient='records')})

# predict using the deployed model
result = service.run(test_sample)
print(result)
#print("The output is:", result.json())

#url = 'http://5af42fb2-502b-4281-8a72-64f5722f362f.southcentralus.azurecontainer.io/score'
# Set the content type
headers = {'Content-type': 'application/json'}

response = requests.post(service.scoring_uri, test_sample, headers=headers)

# Print results from the inference
print(response.text)
