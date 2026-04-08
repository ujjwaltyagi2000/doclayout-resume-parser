
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