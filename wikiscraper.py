import requests
from bs4 import BeautifulSoup
import pandas as pd



def wiki_scraper(wiki_url):
    response = requests.get(
        #     url='https://en.wikipedia.org/wiki/List_of_school_shootings_in_the_United_States'
        url=wiki_url
    )
    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'lxml')

        # extract pure data
        data = []
        column_names = []
        # find all sortabel tables
        for i in soup.find_all(name='table'):

            # find all column names
            for j in i.find_all(name='th'):
                column_names.append(j.text.rstrip())

            # find each row
            for k in i.find_all(name='tr'):

                each_row = {}
                # each item in the row
                for index, item in enumerate(k.find_all(name='td')):
                    each_row[column_names[index]] = item.text.rstrip()

                data.append(each_row)

        df = pd.DataFrame(data)

        df = df.dropna(axis=0).reset_index(drop=True)

        return df