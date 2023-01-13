

import argparse


def pre_process_parser():
 parser = argparse.ArgumentParser()

 parser.add_argument('--candle_type', choices=['candle', 'ohlc'])
 return parser