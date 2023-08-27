# Nạp các gói thư viện NumPy và Pandas
import pandas as pd
import numpy as np
import regex
import string
import os

# Cai dat config cho underthesea

from underthesea import word_tokenize, pos_tag, sent_tokenize

files = './files/'
##LOAD EMOJICON
file = open(files+'emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open(files+'teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open(files+'english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open(files+'wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open(files+'vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '                    
    document = new_sentence  
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()


def process_postag_thesea(text):
    text = text+' a'
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','').lower()
        ###### POS tag
        lst_word_type = ['V', 'N', 'A'] 
        # A: Adj, AB: Abverb, V: verb, VB: verb in base form, R: Relative abverb (when, why, where, how; trong tiếng việt là khi nào, tại sao...)
        sentence = ' '.join( word[0].lower() if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document


def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_stockword(text, stockword):
    ###### REMOVE stop words
    document = ' '.join('' if word.upper() in stockword else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_time(text):
    ###### REMOVE stop words
    document = ' '.join('' if word.find('/')!=-1 else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_timeword(text):
    ###### REMOVE stop words
    document = ' '.join('' if word in ['ngày', 'tháng', 'năm', 'quý', 'lần'] else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_number(text):
    ###### REMOVE stop words
    document = ' '.join('' if any(x.isdigit() for x in word) else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_dontu(text):
    ###### REMOVE stop words
    document = ' '.join('' if word.find('_')<1 or word.find('_')>=len(word)-1 else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def clean_text(text, stockword_list = ['ABC', 'ACM', 'AGF', 'AMD', 'AMP', 'APT', 'ASA', 'AST', 'ATA', 'ATB', 'ATG', 'BHC', 'BHT', 'BT6', 'C12', 'CDO', 'CEG', 'CIG', 'CLG', 'CMI', 'CPI', 'CTA', 'CYC', 'DCT', 'DDM', 'DHB', 'DIC', 'DLG', 'DLR', 'DP2', 'DXV', 'FLC', 'GAB', 'GGG', 'HAG', 'HAI', 'HAS', 'HFX', 'HKB', 'HLG', 'HLY', 'HNG', 'HNM', 'HOT', 'HPI', 'HSA', 'HSI', 'HTT', 'HU1', 'HU3', 'HVG', 'HVN', 'ISG', 'ITA', 'JOS', 'JVC', 'KAC', 'KHL', 'KSH', 'LCC', 'LCM', 'LHG', 'LM3', 'LO5', 'LQN', 'LTC', 'MCG', 'MCI', 'MES', 'MHC', 'MPT', 'MSN', 'MWG', 'NDF', 'NET', 'NOS', 'NSC', 'NSG', 'NT2', 'NVT', 'OGC', 'ONW', 'OPC', 'PAC', 'PAN', 'PBP', 'PC1', 'PCE', 'PCT', 'PET', 'PGC', 'PGD', 'PGS', 'PHH', 'PIT', 'PIV', 'PLC', 'PLO', 'PLX', 'PMB', 'PMG', 'PNJ', 'POM', 'POW', 'PPC', 'PRT', 'PSD', 'PTE', 'PTL', 'PVA', 'PVB', 'PVC', 'PVD', 'PVE', 'PVG', 'PVH', 'PVR', 'PVS', 'PVT', 'PVV', 'PVX', 'PVY', 'PXA', 'PXI', 'PXM', 'PXS', 'PXT', 'QBS', 'RCD', 'RDP', 'RIC', 'S12', 'S27', 'SAB', 'SAM', 'SBT', 'SBV', 'SCD', 'SCO', 'SD1', 'SD5', 'SD6', 'SDJ', 'SDP', 'SDY', 'SGO', 'SGT', 'SII', 'SJC', 'SJD', 'SMA', 'SP2', 'STT', 'TBH', 'TCJ', 'TCK', 'TCR', 'TDH', 'TGG', 'TH1', 'TNI', 'TNM', 'TOP', 'TS4', 'TSD', 'TTF', 'UDC', 'V11', 'VAT', 'VC5', 'VCT', 'VES', 'VFG', 'VIR', 'VNI', 'VOS', 'VPC', 'VSF', 'VST', 'VTG', 'VTI', 'VTR', 'VVN', 'X77', 'XPH']):
    document = text#process_text(text, emoji_dict, teen_dict, wrong_lst)
    document = covert_unicode(document)
#     document = process_special_word(document)
    document = process_postag_thesea(document)
    document = remove_stockword(document, stockword_list)
    document = remove_number(document)
    document = remove_timeword(document)
    document = remove_time(document)
    # document = remove_dontu(document)
    return document
