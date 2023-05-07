from datasets import load_csv
if __name__ == '__main__':
    data = load_csv(['WWE', 'ZION'], ['Date', 'Open', 'Close'], '2002-10-31',
                    '2002-11-04')
    print(data.shape)
    print(data)