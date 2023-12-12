from pandas_datareader import data as pdr

import yfinance as yf
import numpy as np

yf.pdr_override() # <== that's all it takes :-)


def GenerateOutput_1(df_1):

    df_1["Output"] = np.nan


    n = 5
    for i in range(n,df.shape[0]-n):
        df_1['Output'].iloc[i] = 0
        OutputCriteria = (max(df_1["High"].iloc[i+1],df_1["High"].iloc[i+2],df_1["High"].iloc[i+3],df_1["High"].iloc[i+4],df_1["High"].iloc[i+5]) - df_1["High"].iloc[i]) / df_1["High"].iloc[i]

        if(OutputCriteria > .1):
            df_1['Output'].iloc[i] = 1
    return df_1

def CalculateRSI(df_1):
    df_1['change'] = df['Close'].diff(1) # Calculate change
    df_1['change']=df['change'].apply(lambda x:round(x,2))

    df['gain'] = np.select([df['change']>0, df['change'].isna()], 
                       [df['change'], np.nan], 
                       default=0) 
    
    df_1['gain']=df['gain'].apply(lambda x:round(x,2))

    df['loss'] = np.select([df['change']<0, df['change'].isna()], 
                       [-df['change'], np.nan], 
                       default=0)
    df_1['loss']=df['loss'].apply(lambda x:round(x,2))



    # create avg_gain /  avg_loss columns with all nan
    df_1['avg_gain'] = np.nan 
    df_1['avg_loss'] = np.nan

    n = 14 # what is the window

    # keep first occurrence of rolling mean
    df_1['avg_gain'][n] = df_1['gain'].rolling(window=n).mean().dropna().iloc[0] 
    df_1['avg_loss'][n] = df_1['loss'].rolling(window=n).mean().dropna().iloc[0]

    # This is not a pandas way, looping through the pandas series, but it does what you need
    for i in range(n+1, df.shape[0]):
        df_1['avg_gain'].iloc[i] = (df_1['avg_gain'].iloc[i-1] * (n - 1) + df_1['gain'].iloc[i]) / n
        df_1['avg_loss'].iloc[i] = (df_1['avg_loss'].iloc[i-1] * (n - 1) + df_1['loss'].iloc[i]) / n

    # calculate rs and rsi
    df_1['rs'] = df_1['avg_gain'] / df_1['avg_loss']
    df_1['rsi'] = 100 - (100 / (1 + df_1['rs'] ))

    df_1['rs']=df['rs'].apply(lambda x:round(x,2))
    df_1['rsi']=df['rsi'].apply(lambda x:round(x,2))

    return df_1

def GetSubDF(df_1):
    subClass = ["Open","Close","High","Low","Volume","macd","macd_h","macd_s","rs","rsi"]

    df_1 = df_1[subClass]
    return df_1


def CalculateSMA(df_1,TimeFrame=50):

    df_1["SMA_"+str(TimeFrame)] = df_1['Close'].rolling(TimeFrame).mean()
    df_1["SMA_"+str(TimeFrame)] = df["SMA_"+str(TimeFrame)].apply(lambda x:round(x,2))
    return df_1

def RoundDFValues(df_1):
    df['Open']=df['Open'].apply(lambda x:round(x,2))
    df['High']=df['High'].apply(lambda x:round(x,2))

    df['Low']=df['Low'].apply(lambda x:round(x,2))
    df['Close']=df['Close'].apply(lambda x:round(x,2))


    df['Adj Close']=df['Adj Close'].apply(lambda x:round(x,2))
    df['Volume']=df['Volume'].apply(lambda x:round(x,2))


    df['macd']=df['macd'].apply(lambda x:round(x,2))
    df['macd_h']=df['macd_h'].apply(lambda x:round(x,2))

    df['macd_s']=df['macd_s'].apply(lambda x:round(x,2))


    return df_1





if __name__=="__main__":
    symbol = "ITC.NS"#"TATAMOTORS.NS"
    df = pdr.get_data_yahoo(symbol, start="2013-01-01", end="2023-12-12")


    # Get the 26-day EMA of the closing price
    k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    # Get the 12-day EMA of the closing price
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d
    # Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s
    # Add all of our new values for the MACD to the dataframe
    df['macd'] = df.index.map(macd)
    df['macd_h'] = df.index.map(macd_h)
    df['macd_s'] = df.index.map(macd_s)

    #Date,Open,High,Low,Close,Adj Close,Volume,macd,macd_h,macd_s
    df = RoundDFValues(df)

    df = CalculateRSI(df)

    df = GetSubDF(df)

    df = CalculateSMA(df,50)

    df = CalculateSMA(df,200)

    df = GenerateOutput_1(df)

    csvname = symbol+".csv"
    df.to_csv(csvname)