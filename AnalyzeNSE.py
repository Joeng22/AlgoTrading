import nsepython
from NSEData import *
if __name__=="__main__":
    symbols=nsepython.nse_eq_symbols()
    print(symbols)

    for each_sy in symbols:
        try:        
            df = Calculate_Matrics(each_sy)
            Save_CSV(df,each_sy,"NSE_CSV/")
        except Exception as e:
            print("The error is: ",e)
