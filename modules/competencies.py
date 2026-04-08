from config.settings import *
import en_core_web_sm
nlp = en_core_web_sm.load()
import pandas as pd
import re

def get_bullets(text):
            
    try:
        pattern1 = r"[^\w][\s]\d{1,2}\)+"
        pattern2 =  r"\s?[0-9]\.\s+"
        pattern3 =  r"\s[a-z]\)\s+"
        pattern4 =  r"[^\w]•\s+|●\s+|✓\s+|▪\s+|❖\s+|➢\s+|[^\w]\*\s+|"
    
        pattern5 =  r"\s[A-Z]\)\s+"
        pattern6 =   r"[^\w][IVX]+\.\s+" 
        finalBulletSet = set()
        finalBullet = set()

        bullets1 = re.findall(pattern1,text)
        if len(bullets1) > 0:
            finalBulletSet.add(bullets1[0])
        


        bullets2 = re.findall(pattern2,text)
        if len(bullets2) > 0:
            finalBulletSet.add(bullets2[0].strip())
        bullets3 = re.findall(pattern3,text)
        if len(bullets3) > 0:
            finalBulletSet.add(bullets3[0])

        bullets4 = re.findall(pattern4,text)

        bulletSet = set(bullets4)
        finalBulletSet = finalBulletSet | bulletSet
        
        

        bullets5 = re.findall(pattern5,text)
        if len(bullets5) > 0:
            finalBulletSet.add(bullets5[0])

        bullets6 = re.findall(pattern6,text)
        if len(bullets6) > 0:
            finalBulletSet.add(bullets6[0])

        for x in finalBulletSet:
            s = x.replace('\n','')
            z = s.replace('\x0c','')
            finalBullet.add(z.replace(' ',''))
            
        newList = list(finalBullet) 
        updatedList = [x for x in newList if x != ""]
        finalBullet1 = set(updatedList)
        Bullets_Total = len(finalBullet1) 
        flag_Bullet=1
        if Bullets_Total > 4 or Bullets_Total==0:
            flag_Bullet = 0
        

        else:
            for i in finalBullet1:
                if i == '●' or i == '•' or i == '▪':
                    continue
                else:
                    flag_Bullet = 0
                    break


    
        return list(finalBullet1),Bullets_Total,flag_Bullet
    except Exception as e :
        print("get_bullets ",e)
        return [],0,0
    
def getATS_dates(text):
    try:
        
        z = []
        ats = []
        r = []
        x = []
        n = []
        all6 = re.findall(r"([\d]{4}\s?[-|—|–]\s?[\d]{4}[^0-9])",text)
        for i in all6:
            n1 = i.replace('(','')
            n2 = n1.replace(')','')
            n2 = n2.strip()
            n.append(n2)


        all8  = re.findall(r"\b\d{1,2}[/]\d{4}\b\s?[-|–|—]\s?\b\d{1,2}[/]\d{4}\b", text)

        all2 = re.findall(r"\b(\d{2}/\s?\d{4}) [-|–|—] (Present|Current|till date)\b", text, re.IGNORECASE)
        
        for i in all2:
            if i[1] not in i[0]: 
                u = i[0]+' - '+i[1]
                r.append(u)
            else:
                for j in i:
                    r.append(j) 
                    break  

        z = re.findall(r'(?<![\d-])(?:(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{4})(?!\s*[-–—]\s*(?:PRESENT|present|CURRENT|current|TILL DATE|till date))(?:\s*[-–—]\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{4})?(?![\d-])', text,re.IGNORECASE)

    
        all = re.findall(r"\b([\w]{3} \d{4} [-–—] (PRESENT|CURRENT|TILL DATE))\b", text, re.IGNORECASE)
        
        if len(all) > 0:
            for i in all:
                if i[1] not in i[0]:
                    a = i[0]+' - '+i[1]
                    x.append(a)    
                else:
                    for j in i:
                        x.append(j) 
                        break  

    
        
        all10 =  re.findall(r"(\d{2}[/]\d{2}[/]\d{2,4}\s[–|-|—]\s\d{2}[/]\d{2}[/]\d{2,4})",text)
        ats = n+all8+z+all10+r+x
        new_ats=[]
        for i in ats:
            i=i.replace("\n"," ")
            i=i.strip()
            new_ats.append(i)
        return list(new_ats)
    
    except Exception as e:
        print("getATS_dates ",e)
        return []

#     #Extracting Non-ATS  dates 
def get_nonATSdates(text):
    
    try: 
        
        ats = getATS_dates(text)
        t = []
        for i in ats:
            if '-' in i:
                x = i.split("-")
                for j in x:
                    t.append(j.strip())  
            elif '/' in i:
                x = i.split("-")
                for j in x:
                    t.append(j.strip()) 
            elif '/' in i:
                x = i.split("–")
                for j in x:
                    t.append(j.strip())  
        

        # 01–01–2002
        all10 =  re.findall(r"(\d{2}[/]\d{2}[/]\d{2,4}\s[–-—]\s\d{2}[/]\d{2}[/]\d{2,4})",text)
        if len(all10) == 0:
            al = re.findall(r"\b\d{1,2}\s?[/]\s?\d{1,2}\s?[/]\s?\d{4}\b", text)

        all = re.findall(r"\b\d{1,2}\s?[–-]\s?\d{1,2}\s?[–-]\s?\d{4}\b", text)

        # all_ = re.findall(r"\d{1,2}\s[/–]\s\d{1,2}\s[/–]\s\d{4}",text)

        # 2004–11–11
        all1 =  re.findall(r"\b(?!.*\d{1,2}\s?[-–/]\s?\d{1,2}\s?[-–/]\s?\d{4})\d{4}\s?[-–/]\s?\d{1,2}\s?[-–/]\s?\d{1,2}\b",text)
        # all1_ = re.findall(r"\d{4}\s?[/–-]\s?\d{1,2}\s?[/–-]\s?\d{1,2}",text)

        # 20 11 2011
        all2 = re.findall(r"\d{1,2}\s?(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text, re.IGNORECASE)
    
        
    
        # May 08, 2004
        all3 = re.findall(r"\b((JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s[\d]{1,2}[,]\s[\d]{4})\b",text)
        all3_ = [] 
        if len(all3) > 0:
            for i in all3:
                all3_.append(i[0])

    
        
        
        # 08 June, 2004
        all4 = re.findall(r"([\d]{1,2}\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[,]\s[\d]{4})",text)
        all4_ = [] 
        if len(all4) > 0:
            for i in all4:
                all4_.append(i[0])
        
        # 2022, January 12
        all5 = re.findall(r"([\d]{4}[,]\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s[\d]{1,2})",text)  
        all5_ = [] 
        if len(all5) > 0:
            for i in all5:
                all5_.append(i[0])
        
    
        all7 = re.findall(r"((?<!\S)(?!.-.\b)(?:JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[,]\s[\d]{4})(?!\S)", text)



        all7_ = [] 
        if len(all7) > 0:
            for i in all7:
                all7_.append(i[0])

        # 14 2004
        new1 =  re.findall(r"(\b\d{1,2}\s\d{4})\b", text)
    
        # 01/2002 or 01-2002 or 01.2020
        new2 = re.findall(r"(?<!\d\d/)(?<!\d)(?!\d{2}/\d{2}/\d{4})([\d]{1,2}[.–/][\d]{4})(?=[^0-9]|$)(?<!\d\d)",text)
        
    

        new21 = [ele for ele in new2 if ele not in t]
    
        # 2022/05
        new3 = re.findall(r"([\d]{4}[/][\d]{1,2})(?=[^0-9\s]|$)",text)

        #  2022/05 to 2022/06
        new4 = re.findall(r"(\d{2,4}[/]\d{1,2}\s[t][o]\s\d{2,4}[/]\d{1,2})(?=[^0-9\s]|$)",text)


        new5 = re.findall(r"\d{1,2}[\s?][-](JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s?][-][\d]{4}",text)
        new5_ = [] 
        if len(new5) == 0:
            new51_= re.findall(r"((JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?[-]\s?[\d]{4})", text,re.IGNORECASE)
        
        if len(new51_) > 0:
            for i in new51_:
                new5_.append(i[0])
            
        
        
        new7 = re.findall(r"\b(\d{1,2}[\s]?[/–-][\s]?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s]?[/–-][\s]?\d{4})\b", text,re.IGNORECASE)
        new7_ = [] 
        if len(new7) > 0:
            for i in new7:
                new7_.append(i[0])
    
    
        new6 = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", text) 
        res = al + all + all1  + all2 + all3_ + all4_ + all5_ + all7_ + new1 + new21 + new3 + new4+ new7_+ new5 + new5_
        res1 = []
        if len(res) == 0 and len(ats) == 0:
            res1 = res + new6
        else:
            res1 = res   

                
        return list(res1)
    except Exception as e:
        print("get_nonATSdates ",e)
        return []
    
def get_namedEntityMeasurable(text):
    try:
        print("here at mesaurableee")
        mesurable=[]
        new_textx = re.findall(r"([\d]{1,4}(million|billion|trillion))",text)
        if len(new_textx) > 0:
            for i in new_textx:
                mesurable.add(i[0])


        
        #preprocessing data
        text = text.replace('\n'," ")
        text = text.replace('('," ")
        pattern = r"\s[0-9]\)+\s?"
        new_text= re.sub(pattern, "", text)
        pattern1 = r"[0-9][.][a-zA-Z]"
        pattern2 = r"\b[0-9]{1,4}[a-zA-Z]\b"
        pattern3 = r"[a-zA-Z][.][0-9]"
        new_text0 = re.sub(pattern1, "",  new_text)
    
        new_text1= re.sub(pattern2, "",  new_text0)
        new_text2= re.sub(pattern3, "",  new_text1)

        new_text2 = new_text2.replace(')'," ")
        # Process the text with Spacy

        doc = nlp(new_text2)

        # Iterate through named entities
        for entity in doc.ents:
        

            if (entity.label_ == 'PERCENT' or entity.label_ == 'MONEY' or entity.label_ == 'CARDINAL' or entity.label_ == 'ORDINAL' or entity.label_ == 'QUANTITY'):
            
                mesurable.append(entity.text)
        
        return mesurable
    except Exception as e:
        print("get_namedEntityMeasurable ",e)
        measurable=[]
        return measurable

def remove_duplicates(input_list):
    result = []
    for i, s1 in enumerate(input_list):
        found_duplicate = False
        for j, s2 in enumerate(input_list[i+1:], start=i+1):
            if s1 in s2 or s2 in s1:  
                found_duplicate = True
                break
        if not found_duplicate:
            result.append(s1)
    return result



def find_words(input_list, search_list):


    for i in range(len(input_list)):
    
        list_input = [word.rstrip('.') for word in input_list[i:i+len(search_list)]]
        list_input = [word.lstrip('(') for word in list_input]
        list_input = [word.rstrip(')') for word in list_input]
        list_input = [word.rstrip(',') for word in list_input]

        if list_input == search_list:
    
            max_context = min(i, 3) 
            min_context = min(len(input_list) - i - len(search_list), 3) 
            output_string = ""
            

            for j in range(max_context):
                output_string += input_list[i - max_context + j] + " "

            output_string += " ".join(search_list) + " "
            for j in range(min_context):
                output_string += input_list[i + len(search_list) + j] + " "

            return output_string.strip()

    return ""

def get_measurableUpdated(text,finalBullet,measurable,phone_all1,dates):
    try:
    
        finalBullet_list  = list(finalBullet)
        phone_all1_list = list(phone_all1)
        measurable_updated = [x for x in measurable if x not in finalBullet_list]
        measurable_updated = [x for x in measurable_updated if x not in phone_all1_list]
        measurable_updated = [x for x in measurable_updated if x not in dates]
        measurable_updated = [x for x in measurable_updated if "#" not in x]
        measurable_updated = [x for x in measurable_updated if '.com' not in x]
        measurable_updated = [x for x in measurable_updated if '.in' not in x]
        measurable_updated = [x for x in measurable_updated if '.io' not in x]
        
        pattern2 =  r"\s[0-9]{1,2}\.\s+"
        bullets2 = re.findall(pattern2,text)
        bullet2_new = []
        for x in bullets2:
            s = x.replace('\n','')
            z = s.replace('\x0c','')
            bullet2_new.append(z.replace(' ',''))
    
        measurable_updated = [x for x in measurable_updated if x not in bullet2_new]

        list1 = ['1','2','3','4','5','6']
        measurable_updated = [x for x in measurable_updated if x not in list1]
        measurable_updated = [x for x in measurable_updated if '\uf0fc' not in x]
        class1 = ['10th','12th','10TH','12TH']
        measurable_updated_new = []  
    
        for i in measurable_updated:
        
            if i.isdigit() and len(i) >= 6:
        
                continue
            elif '/' in i:
        
                pattern = r"([\d]{1,2}\s?[/]\s?[\d]{4})"
                x = re.findall(pattern,i)
            
                if len(x)>0:
            
                    continue
                else:
                    measurable_updated_new.append(i)
                    continue

            elif i in class1:
        
                continue

            elif len(i) >=  15:
            
                continue
        
            else:
        
                measurable_updated_new.append(i)
    
        measurable_updated_change = []
        # file_path = 'Measurable.xlsx'
        dataframe1 = pd.read_excel(MEASURABLES_EXCEL_FILE_PATH,usecols='A')
        # dataframe1 = pd.read_excel('Named Entity.xlsx',usecols='A')
        SKILLS_DB = dataframe1.values.tolist()
        topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
        topics_flat =  [x.lower() for x in topics_flat1]

        flag = 1
    
        for i in measurable_updated_new:
            for j in i:
                if j.lower() >= 'a' and j.lower() <= 'z':
                    continue
                else:
        
                    flag = 0
                    break
            if flag == 0:
        
                flag = 1
                measurable_updated_change.append(i)
            else:
        
                if i.lower() in topics_flat:
            
                    measurable_updated_change.append(i.lower())

        measurable_updated_change_new = []        
        new_ = re.findall(r"([\d]{1,2}\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))",text)
        new = [] 
        if len(new_) > 0:
            for i in new_:
                new.append(i[0])

        measurable_updated_change_new = [x for x in measurable_updated_change if x not in new]  
        measurable_updated_change_new1 = [elem for elem in measurable_updated_change_new if not re.search(r"[\d]\s?[-–]", elem)]
        new1_ = re.findall(r"\d+\+",text)
        measurable_string = []
        if len(new1_) > 0:
            measurable_updated_change_new1.extend(new1_)
        final_ans=  set(measurable_updated_change_new1)
        
        
        measure_dict = {}
        if len(final_ans) > 0:
            for i in final_ans:
                found_matches = find_words(text.split(),i.split())
                measure_dict[i] = found_matches
                if found_matches:
                    measurable_string.append(found_matches)
        
        measurable_output = remove_duplicates(measurable_string)
        final_measurable = []
        for value in measurable_output:
            for key, val in measure_dict.items():
                if val == value:
                    final_measurable.append(key)
        
        # print(measure_dict)
        # print(final_measurable)
        return list(final_measurable)
    except Exception as e:
        print("Measurable Updated Cleaning",e)
        return []