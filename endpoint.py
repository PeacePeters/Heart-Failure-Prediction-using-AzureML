import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    """Method to bypass the server certificate verification on client side
        
        Args:
            None
            
        Returns:
            None
        """
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://e5bf398f-b130-49f5-893d-8e3c729d8bcb.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = '1fyaYzZZb4v0ouH1azj3iPksgmwxtjur'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "Age":75,
            "anaemia":0,
            "creatinine_phosphokinase":582,
            "diabetes":0,
            "ejection_fraction":20,
            "high_blood_pressure":1,
            "platelets":265000,
            "serum_creatinine":1.9,
            "serum_sodium":130,
            "sex":1,
            "smoking":0,
          },
          {
            "Age":49,
            "anaemia":1,
            "creatinine_phosphokinase":80,
            "diabetes":0,
            "ejection_fraction":30,
            "high_blood_pressure":1,
            "platelets":427000,
            "serum_creatinine":0,
            "serum_sodium":138,
            "sex":0,
            "smoking":0,
          },
        ]
    }

body = str.encode(json.dumps(data))

# URL for the web service
#url = 'http://5af42fb2-502b-4281-8a72-64f5722f362f.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://e5bf398f-b130-49f5-893d-8e3c729d8bcb.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
#key = '1fyaYzZZb4v0ouH1azj3iPksgmwxtjur'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())

req = urllib.request.Request(scoring_url, input_data, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
