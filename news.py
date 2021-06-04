from newsapi import NewsApiClient
import json

# Init
api = NewsApiClient(api_key='a19ad8ae04bb4be49ad15bdd40c99f48')

top = api.get_everything(q="beschränkung", language='de')
print()
print(str(top["totalResults"]) + " Erbegnisse")
print()
print()
print("Beispiel:")
print()
print(top["articles"][0]['title'])
print()
print(top["articles"][0]['content'])
print()
print("Quelle: " + str(top["articles"][0]['source']['name']))
print()
print()



from newsplease import NewsPlease
import os 

article = NewsPlease.from_url('https://www.lra-aoe.de/informationen-zu-corona/dummy')
print("Von news please:")
print()
print(article.title)
print()
print(os.linesep.join(article.maintext.split(os.linesep)[:10]))
print("...")


import requests

#RKI API
response = requests.get("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=1%3D1&outFields=GEN,cases7_per_100k&outSR=4326&f=json")

print()
print()
print("Stadt:\t\t" + str(response.json()["features"][0]["attributes"]["GEN"]))
print("7-Tageinzidenz:\t" + str(response.json()["features"][0]["attributes"]["cases7_per_100k"]))
