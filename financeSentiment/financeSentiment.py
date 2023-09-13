"""This script use for predict financial sentiment impact on stock market"""

import pandas as pd
import numpy as np
import regex
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix,roc_curve,auc
from datetime import datetime, timedelta



from processor.text import TextProcessor
text_processor = TextProcessor()

#Preprocessing NLP
def remove_stockword(text, stockword):
    ###### REMOVE stock words
    document = ' '.join('' if word in stockword else word for word in text.split())
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_time(text):
    ###### REMOVE time format
    document = ' '.join('' if word.find('/')!=-1 else word for word in text.split())
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def remove_timeword(text):
    ###### REMOVE time words
    document = ' '.join('' if word in ['ngày', 'tháng', 'năm', 'quý', 'lần'] else word for word in text.split())
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def clean_text(text, 
               stockword_list=None
               ):
    # Chuyển tất cả chữ trong câu thành chữ thường
    sentence = text.lower()
    
    # Loại bỏ các ký tự số hoặc dấu câu
    sentence = text_processor.remove_punctuation_number(sentence)
    
    # Chuẩn hoá sang unicode
    sentence = text_processor.covert_unicode(sentence)
    
    # Xử lý từ ghép
    sentence = text_processor.process_postag_thesea(sentence,
                                                    lst_word_type = ['A', 'AB', 'V', 'VB', 'VY', 'R', 'N'])
    
    # Xử lý từ có không
    sentence = text_processor.process_special_word(sentence)
    
    if stockword_list!=None:
        sentence = remove_stockword(sentence, stockword_list)
    else:
        pass
    sentence = remove_timeword(sentence)
    sentence = remove_time(sentence)
    return sentence


class Classification():
    def __init__(self, 
                 model_ = None):
        self.model=model_
        self.stockList = None
    
    #Clean text data
    def clean_text(self, 
                   text,
                   stockword_list):
        return clean_text(text, stockword_list=stockword_list)
    #Vectorize text
    def Word2Vec(self, 
                 text_array,
                 method = CountVectorizer()):
        vectorizer = CountVectorizer().fit(text_array)
        self.vectorizer = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        self.list_text = feature_names
        text_vec = vectorizer.transform(text_array)
        sentiments = pd.DataFrame(data=text_vec.toarray(), columns = feature_names)
        return sentiments
        # else:
        #     raise Exception("Need to import "+str(method)+", set config and try again!")
    #Build and fit model
    def financeSentiment(self, 
                         dataframe,
                         method = CountVectorizer(),
                         algorithm = RandomForestClassifier()):
        columns = ["date", "stock", "title", "hour", "open_price", "close_price"]
        if any(columns!=dataframe.columns):
            raise Exception("Need to prepare dataset follow template "+ str(columns))
        else:
            dfRoot = dataframe
            dfRoot["hour"] = dfRoot["hour"].apply(lambda x:float(x))
            title = dfRoot[["date", "stock", "title", "hour"]]
            price = dfRoot[["date", "stock", "open_price", "close_price"]]
            
            #Media
            stockword_list = [i.lower() for i in dfRoot["stock"].unique().tolist()]
            self.stockList = stockword_list
            title["title_new"] = title["title"].apply(lambda x:self.clean_text(x, stockword_list=stockword_list))
            title['weekday'] = title['date'].dt.day_name()
            title['date'][title['weekday']=='Sunday'] = title[title['weekday']=='Sunday']['date'] - timedelta(days=2)
            title['date'][title['weekday']=='Saturday'] = title[title['weekday']=='Saturday']['date'] - timedelta(days=1)
            title['hour'][title['weekday']=='Sunday'] = 19
            title['hour'][title['weekday']=='Saturday'] = 19
            title['new_date'] = title['date']
            title['weekend'] = False
            title['weekend'][title['weekday']=='Friday'] = title[title['weekday']=='Friday']['hour']>16
            title['new_date'][title['hour']>16] = title[title['hour']>16]['new_date']+timedelta(days=1)
            title['new_date'][title['weekend']==True] = title[title['weekend']==True]['new_date']+timedelta(days=2)
            title['session'] = ['return_in_session' if (x>8) and (x<16) else 'return_before_session' for x in title['hour']]
            
            #Stock
            price['close_price_lag'] = price.groupby('stock')['close_price'].shift()
            price['open_price_next'] = price.groupby('stock')['open_price'].shift(-1)
            price = price.dropna()
            price['return_in_session'] = np.log(price['close_price']/price['open_price'])
            price['return_before_session'] = np.log(price['open_price']/price['close_price_lag'])
            price_stack = price[['date', 'stock', 'return_in_session', 'return_before_session']].set_index(['date', 'stock']).stack().reset_index()
            price_stack.columns = ['date', 'stock', 'session', 'return']
            price_stack = price_stack.sort_values(['stock', 'date', 'session'])
            price_stack = price_stack[price_stack['return']!=np.inf]
            price_stack = price_stack[price_stack['return']!=-np.inf]
            
            #Merge
            price_stack = price_stack.rename(columns={'date':'new_date'})
            df_delta = title.merge(price_stack, on=['new_date', 'stock', 'session'], how='left')
            df_delta_sub = df_delta
            df_delta_sub['title_new'] = df_delta_sub['title_new']+' '
            df_final = df_delta_sub[['stock', 'title_new', 'new_date', 'session', 'return']].groupby(['new_date', 'stock', 'session', 'return']).sum().reset_index()
            df_final['title_new'] = df_final['title_new'].str[:-1]
            
            # Loai bo nhung dong return la 0
            df_final = df_final[df_final['return']!=0]
            
            # Tao bien deep de giu lai nhung dong return thuc su co y nghia
            df_final['deep']=0
            df_final['deep'][df_final['return']>df_final['return'].quantile(0.999)]=100
            df_final['deep'][df_final['return']<df_final['return'].quantile(0.001)]=100
            df_final = df_final.reset_index(drop=True)
            
            #Tao bien dummies ma chung khoan
            dummies = pd.get_dummies(df_final['stock'])
            
            #Word2Vec
            X = df_final["title_new"]
            sentiments = self.Word2Vec(X)
            sentiments[dummies.columns] = dummies
            
            #Loc nhung tu co nghia
            sentiments['deep'] = df_final['deep']
            deep = pd.DataFrame(data = sentiments[sentiments['deep']==100].sum(), columns=['deep'])
            deep = deep[deep['deep']>0]
            deep_lst = deep.index.tolist()
            
            # Top những từ xuất hiện nhiều hơn 1000 lần
            big = pd.DataFrame(data=np.count_nonzero(sentiments, axis=0), index=sentiments.columns, columns=['counts'])
            big_lst = big[big['counts']>0.01*len(sentiments)].index.tolist()
            
            for i in deep_lst:
              if i not in big_lst:
                big_lst.append(i)
            
            sentiments_new = sentiments[big_lst]
            sentiments_new['title_new'] = df_final['title_new']
            sentiments_new['return'] = df_final['return']
            sentiments_new['stock'] = df_final['stock']
            sentiments_new['date'] = df_final['new_date']
            sentiments_new['session'] = df_final['session']
            sentiments_new['sum'] = sentiments_new[big_lst].sum(axis=1)
            sentiments_new = sentiments_new[sentiments_new['sum']>sentiments_new['sum'].quantile(0.2)]
                        
            
            df = sentiments_new
            # Tao bien binary cho return
            def get_binary(x):
              if x>=0:
                return 0
              if x<0:
                return 1
            
            df['return'] = df['return'].apply(get_binary)
            
            sentiments = df.columns.tolist()[:]
            for i in ['title_new', 'title', 'return', 'deep', 'stock', 'date', 'sum', 'session']:
              if i in sentiments:
                sentiments.remove(i)
                
            X = df[sentiments]
            y = df['return'].astype('int')
            
            X_train = X
            X_test = X
            y_train = y
            y_test = y
            
            #FITTING THE CLASSIFICATION MODEL using Random Forest
            model=RandomForestClassifier(random_state = 42)
            self.model = model.fit(X_train, y_train.values)
            
            #Predict y value for test dataset
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            
            
            #Predict y value for test dataset
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            
            
            print(classification_report(y_test,y_pred))
            print('Confusion Matrix:',confusion_matrix(y_test, y_pred))
            
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            print('AUC:', roc_auc)
            
            result = pd.DataFrame(data=df[["title_new"]])
            result["negative_prob"] = y_prob
            output = result.merge(title[["title", "title_new"]], on="title_new", how="inner")
            return output[["title", "negative_prob"]]
    def predict_text(self, text):
        clean_text = self.clean_text(text, self.stockList)
        remove = ['kinh_tế', 'thị_trường']
        predict_prob = self.model.predict_proba(pd.DataFrame({i: [(i in clean_text) and (i not in remove)] for i in self.model.feature_names_in_}))
        predict_prob = int(predict_prob[0][1]*100)
        if predict_prob >= 50:
            noti1 = 'Khuyến nghị giá sẽ giảm' 
            noti2 = 'Sắc thái tiêu cực là: '+ str(predict_prob)+ '%'
            print(noti1)
            print(noti2)
            # return (noti1, noti2)
        else:
            noti1 = 'Khuyến nghị giá vẫn ổn định' 
            noti2 = 'Sắc thái tiêu cực là: '+ str(predict_prob)+ '%'
            print(noti1)
            print(noti2)
            # return (noti1, noti2)
    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    