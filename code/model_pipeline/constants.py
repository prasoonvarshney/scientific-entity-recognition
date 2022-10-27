LABEL_STUDIO_URL = "http://ec2-54-175-11-3.compute-1.amazonaws.com:8080"
LABEL_STUDIO_AUTH_HEADER = {'Authorization': 'Token 8c7f4625665248b9ee072b34791137cdc6fcaf18'}
ANNOTATIONS_FILE_PATH = "./data/annotated/"
TRAIN_DATA_FILE_PATH = "./data/created_data_train_test_splits"
TEST_DATA_FILE_PATH = "./data/test/"
TEST_DATA_PARAGRAPHS_FILE = "anlp-sciner-test.txt"
TEST_DATA_SENTENCES_FILE = "anlp-sciner-test-sentences.txt"

TRAIN_PAPERS = {
    "5": "2021.findings-acl.212",
    "6": "P19-1051",
    "9": "P15-2083", 
    "11": "P18-2051", 
    "12": "P16-2007", 
    "13": "2022.coling-1.308", 
    "16": "P17-2066", 
    "18": "2020.coling-main.130", 
    "20": "2021.semeval-1.12", 
    "22": "P19-1020", 
    "23": "2022.acl-long.576", 
    "24": "2021.eacl-main.224", 
    "25": "2021.wanlp-1.15", 
    "27": "2020.emnlp-main.563", 
    "29": "P19-1128", 
    "31": "P15-1070", 
    "29": "P19-1128", 
    "29": "P19-1128", 
    "32": "P15-2085", 
    "33": "P16-1158", 
    "34": "P17-1038", 
    "35": "P18-1085",
    "36": "P18-1182",
    "37": "2020.findings-emnlp.20", 
    "53": "BertyBoy"
}

TEST_PAPERS = {
    "40": "2020.acl-main.169",
    "41": "2020.aacl-main.32",
    "42": "2021.findings-acl.175", 
    "43": "P18-3010",
    "44": "2020.loresmt-1.5", 
    "49": "P16-2056", 
    "51": "2020.ecnlp-1.8", 
    "52": "2022.acl-short.36",
    "56": "2020.eamt-1.30"
}

label2id = {
    'O': 0,
    'B-MethodName': 1,
    'I-MethodName': 2,
    'B-HyperparameterName': 3,
    'I-HyperparameterName': 4,
    'B-HyperparameterValue': 5,
    'I-HyperparameterValue': 6,
    'B-MetricName': 7,
    'I-MetricName': 8,
    'B-MetricValue': 9,
    'I-MetricValue': 10,
    'B-TaskName': 11,
    'I-TaskName': 12,
    'B-DatasetName': 13,
    'I-DatasetName': 14
}


