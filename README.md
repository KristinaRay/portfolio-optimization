# portfolio-optimization


Clone the project using the command line

```
git clone git@github.com:KristinaRay/portfolio-optimization.git
```
```
cd portfolio-optimization
```


## Dependencies

Install dependencies by running

```
pip install -r requirements.txt
```
## Usage

Download historical stock prices S&P500 companies

```
python scrape.py --tickers_num 10 --start_date '2017-01-01'
```
Run ```portfolio.py``` to get insights on optimal investment portfolio

```
python portfolio.py
```
