import os 
import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
from argparse import ArgumentParser
import yfinance # library for financial analysis
import datetime

DATA_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

def fetch_tickers(data_url, tickers_num):
    """ 
    Download the data 
    """
    table_class="wikitable sortable jquery-tablesorter"
    response=requests.get(data_url)
    assert response.status_code == 200, "Connection error"
    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable=soup.find('table',{'class':"wikitable"})
    tickers=pd.read_html(str(indiatable))
    # convert list to dataframe
    tickers=pd.DataFrame(tickers[0])
    if tickers_num > 505:
        print("Please, enter number less than 505")
    return tickers.Symbol.values[:tickers_num]
  #"2017-01-01"  
def create_sample(tickers_list, start_date, end_date=None):
    """
    Fetch data sample and save it
    """
    os.makedirs('data', exist_ok=True)
    print("Downloading the data...")
    if not end_date:
        end_date = datetime.date.today()
    
    stock_data = yfinance.download(list(tickers_list),start_date)['Adj Close']
    print('Creating csv file...') 
    # write DataFrame to a csv file
    stock_data.to_csv('data/stock_prices.csv')
    return stock_data
def main():   
    parser = ArgumentParser()
    parser.add_argument("--tickers_num", required=True, type=int, help='Number of the tickers')
    parser.add_argument("--start_date", required=True, type=str, help='Start date')
    parser.add_argument("--end_date", required=False, type=str, help='End date')
    args = parser.parse_args()
    tickers_list = fetch_tickers(DATA_WIKI_URL, args.tickers_num)
    create_sample(tickers_list, args.start_date, args.end_date)

    print("Done")

if __name__ == "__main__":
    main()