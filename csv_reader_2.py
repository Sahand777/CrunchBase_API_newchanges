
# =============================================================================
# importing libararies and the country code dictionary
import requests
import pandas as pd
import numpy as np
from text_filter import text_filter
import pickle

from tqdm import tqdm
country_code_dict = pickle.load(open('country_code_dict', 'rb'))


# =============================================================================
# =============================================================================
# Reading the Data From CrunchBase

url = "https://api.crunchbase.com/bulk/premium/v4/organizations.csv?user_key=fac30a50475edbc79ca1d0cf08282be5"
# response = urllib.request.urlopen(url)
# s=requests.get(url).content
# organizations = pd.read_csv(io.StringIO(s.decode('utf-8')))

organizations = pd.read_csv(url)
head = organizations.head(10)
print(head)
# =============================================================================




master_organizations = pd.DataFrame()
master_organizations["uuid"] = organizations["uuid"]
master_organizations["Naics Code"] = np.nan
master_organizations["Name"] = organizations["name"]
master_organizations["Description"] = organizations["short_description"]
master_organizations["Phone"] = organizations["phone"]
master_organizations["Website Url"] = organizations["homepage_url"]
master_organizations["Contact Title"] = ''     
master_organizations["Contact Name"] = ''
master_organizations["Contact Phone"] = ''
master_organizations["Contact Email"] = organizations["email"]
master_organizations["Address"] = organizations["address"]
master_organizations["City"] = organizations["city"]
master_organizations["State"] = organizations["region"]
master_organizations["Zip"] = organizations["postal_code"]
master_organizations["Country"] = organizations["country_code"].map(country_code_dict)
master_organizations["Employees"] = organizations["employee_count"]
master_organizations["Revenue"] = organizations["revenue_range"]
master_organizations["Founded"] = organizations["founded_on"]
master_organizations["Services"] = organizations["category_groups_list"]
master_organizations["Facebook Url"] = organizations["facebook_url"]
master_organizations["Linkedin Url"] = organizations["linkedin_url"]
master_organizations["Twitter Url"] = organizations["twitter_url"]
master_organizations["Categories"] = organizations["category_list"]
master_organizations=master_organizations.fillna("")            




loaded_model = pickle.load(open('ML_NAICS', 'rb'))
bow_transformer = pickle.load(open('BOW', 'rb'))
labelencoder = pickle.load(open('label_encoder', 'rb'))

  

for i in tqdm(range(master_organizations.shape[0])):
    new_X = pd.Series(text_filter(pd.Series(master_organizations.at[i, 'Description'] + ' ' + master_organizations.at[i,"Categories"] + ' ' + master_organizations.at[i, "Services"])))
    text_bow_new_X = bow_transformer.transform(new_X)
    master_organizations.at[i, 'Naics Code'] = labelencoder.inverse_transform(loaded_model.predict(text_bow_new_X))[0]

head2 = master_organizations.head(30)
print(head2)
master_organizations.to_csv('master_organizations_csv.csv', index = False)
