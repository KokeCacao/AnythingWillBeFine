import datetime

import torch

TICKER_LIST = [
    'MTCH', 'TFII', 'TOST', 'MASI', 'ARCC', 'AOS', 'CZR', 'EMN', 'MTN', 'BCH',
    'PAG', 'PAC', 'RDY', 'YPF', 'FLEX', 'JHX', 'TPR', 'FNF', 'PFGC', 'CHDN',
    'OVV', 'DAR', 'CCK', 'QRVO', 'FHN', 'MANH', 'OSH', 'RGEN', 'NLY', 'ONON',
    'CLF', 'VST', 'JAZZ', 'UHS', 'AGCO', 'ARMK', 'DINO', 'LECO', 'ASR',
    'LBTYK', 'PATH', 'GGB', 'LBTYA', 'BSMX', 'LBTYB', 'SWAV', 'KNX', 'RGLD',
    'LOGI', 'PAA', 'PNW', 'RGA', 'EQH', 'XPEV', 'RNR', 'ROKU', 'TPG', 'AEG',
    'KEP', 'BKI', 'ALLE', 'OC', 'APLS', 'RRX', 'SNX', 'PNR', 'XRAY', 'FFIV',
    'VIPS', 'SSL', 'DCP', 'USFD', 'ALGM', 'MORN', 'DKNG', 'BBWI', 'LII', 'VFC',
    'ICL', 'BXP', 'BSAC', 'CGNX', 'CHE', 'WSC', 'HII', 'UWMC', 'LEGN', 'BILI',
    'HOOD', 'CASY', 'YMM', 'NRG', 'G', 'BILL', 'TX', 'ALLY', 'RHI', 'PSNY',
    'DLB', 'FRT', 'CX', 'LEA', 'SKM', 'RL', 'PSTG', 'WSM', 'CROX', 'WEX',
    'NNN', 'UNM', 'ESLT', 'ATR', 'PCOR', 'WRK', 'GPK', 'SEIC', 'OGE', 'CAR',
    'CIEN', 'KBR', 'HR', 'ALV', 'DVA', 'WWE', 'CAE', 'DCI', 'IRDM', 'SKX',
    'COLD', 'TXRH', 'OLN', 'EWBC', 'TTEK', 'CLH', 'ORI', 'FBIN', 'MIDD', 'IVZ',
    'UGI', 'PSO', 'EME', 'JEF', 'DBX', 'WHR', 'WOLF', 'NOV', 'HAS', 'AA',
    'AAP', 'KNSL', 'SAIA', 'AR', 'EGP', 'ST', 'OLED', 'CACI', 'MTZ', 'HESM',
    'VOYA', 'PLNT', 'STVN', 'BERY', 'FUTU', 'SBS', 'INSP', 'NVT', 'CIB', 'EDR',
    'WCC', 'CBSH', 'NATI', 'GME', 'NVCR', 'XP', 'PAAS', 'FR', 'FCN', 'ACHC',
    'ITT', 'ARW', 'DOCS', 'DSGX', 'CFLT', 'KRTX', 'SEE', 'CW', 'CFR', 'AXTA',
    'INGR', 'CELH', 'GLOB', 'OHI', 'TPX', 'TOL', 'BYD', 'RRC', 'WMS', 'WBS',
    'JLL', 'NXST', 'EXEL', 'GNTX', 'NYT', 'STN', 'IQ', 'MGY', 'NFE', 'LSTR',
    'RBC', 'NVST', 'LFUS', 'EDU', 'MHK', 'THC', 'VMI', 'ADT', 'MAT', 'AIZ',
    'RCM', 'GWRE', 'KGC', 'BLD', 'WF', 'PR', 'PRI', 'GNRC', 'APP', 'BRX',
    'PII', 'AN', 'SF', 'LAD', 'TIMB', 'CWEN', 'FSV', 'MTDR', 'CLVT', 'EHC',
    'CHH', 'NYCB', 'MUR', 'CIG', 'LEVI', 'SITE', 'FOUR', 'MDU', 'QDEL', 'GRFS',
    'VVV', 'GXO', 'TXG', 'SBSW', 'ADC', 'WFG', 'RLI', 'OGN', 'PDCE', 'AQN',
    'SMAR', 'KT', 'CHRD', 'NSA', 'AQUA', 'FTI', 'HLI', 'AMKR', 'SIM', 'X',
    'DOOO', 'SON', 'MEDP', 'LNTH', 'NTRA', 'FAF', 'CACC', 'GMED', 'STAG',
    'CIVI', 'SIGI', 'BLCO', 'SAIC', 'SWN', 'MNDY', 'CSAN', 'WH', 'REYN',
    'NTNX', 'PLUG', 'BC', 'BWXT', 'AGNC', 'SLGN', 'BEPC', 'GIL', 'MUSA', 'FLO',
    'DXC', 'CMA', 'NCLH', 'OPCH', 'CNXC', 'NVEI', 'ALTR', 'MKSI', 'CHX',
    'ROIV', 'IDA', 'STWD', 'CRUS', 'PB', 'HXL', 'SMCI', 'CMC', 'WWD', 'ITCI',
    'TKR', 'ASH', 'BROS', 'LANC', 'MLCO', 'NOVT', 'PVH', 'PPC', 'UNVR', 'HOG',
    'CNM', 'ENSG', 'IPGP', 'COLM', 'MNSO', 'NEP', 'BOKF', 'ALIT', 'ALK',
    'CPRI', 'DUOL', 'PBF', 'TIXT', 'FYBR', 'LNW', 'EEFT', 'SRC', 'WING',
    'TREX', 'SPSC', 'CYBR', 'RMBS', 'SLAB', 'RXDX', 'CCCS', 'EXLS', 'NJR',
    'IGT', 'IONS', 'AIRC', 'ESTC', 'POST', 'WIX', 'WTS', 'NFG', 'RH', 'TRNO',
    'HCP', 'AGI', 'MPW', 'NE', 'NWL', 'SSB', 'HTZ', 'FSK', 'OSK', 'WK', 'DDS',
    'HRB', 'TENB', 'MSA', 'ORA', 'AIT', 'BIPC', 'DRVN', 'EXP', 'FIVN', 'HGV',
    'ASO', 'ATKR', 'AM', 'DNB', 'AAON', 'AYI', 'HGTY', 'ENLC', 'RHP', 'GTLB',
    'EXPO', 'M', 'COKE', 'DV', 'VNOM', 'HALO', 'AMG', 'NEWR', 'TNET', 'MMS',
    'ATI', 'HUN', 'PHI', 'VAL', 'VAC', 'PRGO', 'APG', 'ORCC', 'UFPI', 'HQY',
    'RYN', 'SHC', 'GLBE', 'CWAN', 'FOXF', 'MP', 'NSIT', 'MSM', 'PSN', 'GTLS',
    'RIG', 'DTM', 'TSEM', 'FIZZ', 'SOFI', 'S', 'LU', 'ALKS', 'IART', 'VRT',
    'DISH', 'MSGS', 'SNDR', 'DEN', 'QLYS', 'AXS', 'PENN', 'UBSI', 'WEN', 'FIX',
    'ELAN', 'THG', 'COLB', 'SSD', 'NSP', 'SGRY', 'MTSI', 'ASND', 'POWI', 'POR',
    'OGS', 'BTG', 'INFA', 'AJRD', 'SID', 'KRG', 'APPF', 'CR', 'PROK', 'GOL',
    'IPAR', 'OMF', 'EVR', 'IAC', 'CC', 'ELF', 'BRBR', 'ESI', 'COHR', 'BFAM',
    'ZION', 'FCFS', 'ABG', 'GBCI', 'JHG', 'CXT', 'ESNT', 'SEB', 'WNS', 'MMSI',
    'BKH', 'FFIN', 'MTH', 'WTFC', 'OMAB', 'RLX', 'VLY', 'SQSP', 'HOMB', 'AL',
    'THO', 'SNV', 'SWX', 'HE', 'SMG', 'WFRD', 'FLS'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.001
EPOCHS = 4
BATCH_SIZE = 64
# 32: 157it/sec -> 5024 it/sec
# 2**13: 1.25/it -> 6553.6 it/sec
# 2**12: 1.62/it

EMBEDDING_SIZE = 8
WINDOW_SIZE = 64
N_TEST = 128
assert N_TEST > BATCH_SIZE

START_DATE = datetime.datetime(2002, 1, 1)
END_DATE = datetime.datetime(2023, 3, 31)
# FORMAT = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
FORMAT = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Rating', 'RatingScore',
    'ratingDetailsDCFScore', 'ratingDetailsROEScore', 'ratingDetailsROAScore',
    'ratingDetailsDEScore', 'ratingDetailsPEScore', 'ratingDetailsPBScore'
]
PRED_START_DATE = datetime.datetime(2023, 3,
                                    31) - datetime.timedelta(days=WINDOW_SIZE)
PRED_END_DATE = datetime.datetime(2023, 4, 3)

PERPLEXITY = len(TICKER_LIST) // 10
