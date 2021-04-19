import requests
import wikipedia

obama = wikipedia.WikipediaPage(title='Barack Obama').summary

# Base URL for Spotlight API
annotate_url = "http://api.dbpedia-spotlight.org/en/annotate"

# Base URL for DBPedia resources API
DBPedia_url = f"http://vmdbpedia.informatik.uni-leipzig.de:8080/api/1.0.0/values"

# Annotating all persons from the text about Barack Obama from Wikipedia
params = {
    "text": obama,
    "confidence": 0.5,
    "types": "DBpedia:Person"}
# Response content type
headers = {'accept': 'application/json'}

# GET Request
res = requests.get(annotate_url, params=params, headers=headers).json()
persons = res['Resources']
names = []
for person in persons:
    name = person['@URI'].split('/')[-1] # Finding the end of URIs for all persons
    names.append(name)
names = ",".join(names)

# Parameters for finding more info about each person, birth date and birth name
db_params = {
    "entities": names,
    "property": ["dbo:birthDate", "dbo:birthName"],
    "pretty": "NONE",
    "limit": "100",
    "key": "1234",
    "offset": "0",
    "oldversion": "true"
}
# GET request to DBPedia
db_res = requests.get(DBPedia_url, params=db_params, headers=headers).json()
results = db_res['results']['bindings']
for result in results:
    # Filter out those without the wanted resources.
    if 'dbobirthName' in result.keys() and 'dbobirthDate' in result.keys():
        birthname = result['dbobirthName']['value']
        bday = result['dbobirthDate']['value']
        print(f"{birthname} was born {bday}")


