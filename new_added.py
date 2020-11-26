
# %% Importing Libraries
import requests
import json
import pandas as pd
from pandas.io.json import json_normalize 
from operator import itemgetter
import numpy as np
import pickle
from text_filter import text_filter
import datetime
import time
from tqdm import tqdm


# %% reading the last version of data
master_organizations = pd.read_csv('master_organizations_csv.csv')

# %% Definning functions for counting the number of companies and extracting data and saving it as a pandas data frame.
def company_count(query, url):
    r = requests.post(url, params = userkey , json = query)
    result = json.loads(r.text)
    total_companies = result["count"]
    return total_companies
def url_extraction(query, url):
    r = requests.post(url, params = userkey , json = query)
    result = json.loads(r.text)
    normalized_raw = json_normalize(result['entities'])
    return normalized_raw

# %% Query for getting all of organizations that are updated yesterday
userkey = {"user_key":"user key"}
yesterday = datetime.date.today() - datetime.timedelta(days=1)
yesterday = yesterday.strftime("%m/%d/%Y")
query_organizations_added = {
"field_ids": [
"acquirer_identifier",
"identifier",
"location_identifiers",
"short_description",
"categories",
"num_employees_enum",
"revenue_range",
"operating_status",
"website",
"linkedin",
"facebook",
"twitter",
"phone_number",
"contact_email",
"name",
"revenue_range",
"founded_on",
"created_at"
],
"limit": 1000,
"query": [
{
"type": "predicate",
"field_id": "created_at",
"operator_id": "gte",
"values": [
yesterday        
]
}
]
}
# %% Getting all organizations entity data by looping  
url = "https://api.crunchbase.com/api/v4/searches/organizations"    
raw_organizations = pd.DataFrame()
comp_count = company_count(query_organizations_added, url)
print(comp_count) 
data_acq = 0
# data_acq
while data_acq < comp_count:
    if data_acq != 0:
        last_uuid = raw_organizations.uuid[len(raw_organizations.uuid)-1]
        query_organizations_added["after_id"] = last_uuid
        raw_organizations = raw_organizations.append(url_extraction(query_organizations_added, url),ignore_index=True)
        data_acq = len(raw_organizations.uuid)
    else:
        if "after_id" in query_organizations_added:
            query_organizations_added = query_organizations_added.pop("after_id")
            raw_organizations = raw_organizations.append(url_extraction(query_organizations_added, url),ignore_index=True)
            data_acq = len(raw_organizations.uuid)
        else:
            raw_organizations = raw_organizations.append(url_extraction(query_organizations_added, url),ignore_index=True)
            data_acq = len(raw_organizations.uuid)
    print(data_acq)
print("Extracting Data from Organization is done")
#%% Cleaning the data and changing it to our tamplate
revenue_range = {
"r_00000000": "Less than $1M",
"r_00001000": "$1M to $10M",
"r_00010000": "$10M to $50M",
"r_00050000": "$50M to $100M",
"r_00100000": "$100M to $500M",
"r_00500000": "$500M to $1B",
"r_01000000": "$1B to $10B",
"r_10000000": "$10B+"}
employee_range = {
"c_00001_00010": "1-10",
"c_00011_00050": "11-50",
"c_00051_00100": "51-100",
"c_00101_00250": "101-250",
"c_00251_00500": "251-500",
"c_00501_01000": "501-1000",
"c_01001_05000": "1001-5000",
"c_05001_10000": "5001-10000",
"c_10001_max": "10001+"}
new_added = pd.DataFrame()
new_added["uuid"] = raw_organizations["uuid"]
new_added["Naics Code"] = np.nan
new_added["Name"] = raw_organizations["properties.identifier.value"]
new_added["Description"] = raw_organizations["properties.short_description"]
new_added["Phone"] = raw_organizations["properties.phone_number"]
new_added["Website Url"] = raw_organizations["properties.website.value"]
new_added["Contact Title"] = ''     
new_added["Contact Name"] = ''
new_added["Contact Phone"] = ''
new_added["Contact Email"] = raw_organizations["properties.contact_email"]
raw_organizations["location"] = raw_organizations["properties.location_identifiers"].apply(lambda x: list(map(itemgetter('value'), x)if isinstance(x, list) else ["Not found"])).apply(lambda x : ",".join(map(str, x)))
new_added["Address"] = ''
new_added["City"] = raw_organizations.location.str.split(',', expand=True)[0]
new_added["State"] = raw_organizations.location.str.split(',', expand=True)[1]
new_added["Zip"] = ''
new_added["Country"] = raw_organizations.location.str.split(',', expand=True)[2]
new_added["Employees"] = raw_organizations["properties.num_employees_enum"].map(employee_range)
new_added["Revenue"] = raw_organizations["properties.revenue_range"].map(revenue_range)
new_added["Founded"] = raw_organizations["properties.founded_on.value"]
new_added["Services"] = ''
new_added["Facebook Url"] = raw_organizations["properties.facebook.value"]
new_added["Linkedin Url"] = raw_organizations["properties.linkedin.value"]
new_added["Twitter Url"] = raw_organizations["properties.twitter.value"]
new_added["Categories"] = raw_organizations["properties.categories"].apply(lambda x: list(map(itemgetter('value'), x)if isinstance(x, list) else ["Not found"])).apply(lambda x : ",".join(map(str, x)))
           
#%% adding addresses to our data and removing uuid
for i in tqdm(range(new_added.shape[0])):
    uuid = new_added.at[i, "uuid"]
    link = "https://api.crunchbase.com/api/v4/entities/organizations/"+uuid+"/cards/headquarters_address"
    r = requests.get(link, params = userkey)
    while r.text == 'Usage limit exceeded':
        time.sleep(20)
        r = requests.get(link, params = userkey) 
    normalized_raw = json_normalize(json.loads(r.text))
    if normalized_raw["cards.headquarters_address"][0] != []:        
        address = normalized_raw["cards.headquarters_address"][0][0]
        line_1 = ''
        line_2 = ''
        p_code = ''
        if 'street_1' in address.keys():
            line_1 = address['street_1']
        if 'street_2' in address.keys():
            line_2 = address['street_2']
        if 'postal_code' in address.keys():
            p_code = address['postal_code']
        if line_2 == '':
            new_added.at[i, "Address"] = line_1
        else:
            new_added.at[i, "Address"] = line_1 + ',' + line_2
        new_added.at[i, "Zip"] = p_code

#%% Adding NAICS by using ML

loaded_model = pickle.load(open('ML_NAICS', 'rb'))
bow_transformer = pickle.load(open('BOW', 'rb'))
labelencoder = pickle.load(open('label_encoder', 'rb'))

for i in tdqm(range(new_added.shape[0])):
    new_X = pd.Series(text_filter(pd.Series(new_added.at[i, 'Description'] + ' ' + new_added.at[i,"Categories"] + ' ' + new_added.at[i, "Services"])))
    text_bow_new_X = bow_transformer.transform(new_X)
    new_added.at[i, 'Naics Code'] = labelencoder.inverse_transform(loaded_model.predict(text_bow_new_X))[0] 
new_added.to_csv('new_added.csv', index = False)
pd.concat([master_organizations, new_added]).drop_duplicates().reset_index(drop=True)


#%%
