"""
This script pre-processes data for our stock AI pipeline
"""

import mplfinance as mpf
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import stock_parser
import io

def pre_process(args):
    #import mapt
    # Get the QQQ ticker object
    ticker = yf.Ticker("QQQ")

    # Get the historical market data for the ticker
    df = ticker.history(period="1d", interval='5m') #start='2020-01-23', end='2020-02-23')
    candle_df=df.drop(['Dividends','Stock Splits','Capital Gains'], axis='columns')

    #https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb :: For style info
    # First we set the kwargs that we will use for all of these examples:
    kwargs = dict(type=args.candle_type,figratio=(10,10),figscale=0.85)#volume=True, #type='ohlc'
    # mc = mpf.make_marketcolors(up='white',down='white',edge='black',
    #                         wick={'up':'white','down':'white'}, alpha=1.0)


    mc=mpf.make_marketcolors(
        up='g',
        down='r',
        edge={'up':'black','down':'black'},
        wick={'up':'black','down':'black'},
        ohlc='black',
        alpha=1.0
    )

    s  = mpf.make_mpf_style(marketcolors=mc, facecolor='white', gridstyle = '', figcolor='black')

    buf = io.BytesIO()
    mpf.plot(df,**kwargs,savefig=dict(fname=buf),
        style = s,
        title = '',
        ylabel = '',
        ylabel_lower = '',
        volume = False)
    
    # Opens a image in RGB mode
    #im = Image.open("testsave_SCRIPT.png")
    im = Image.open(buf)#.convert('L')

    #im = im.filter(ImageFilter.FIND_EDGES)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    # Cropped image of above dimension
    # (It will not change original image)
    #(left=0, upper=0, right=width,lower=height) 
    #im1 = im.crop( (122,59,width, height))
    #im1 = im.crop( (121,58,width-66, height-87))
    im.save('testsave_script_cropped.png')

if __name__=='__main__':
    parser=stock_parser.pre_process_parser()
    args=parser.parse_args()
    pre_process(args)
    




