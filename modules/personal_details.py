import re
import phonenumbers

def get_phones(text):
    try:
        print("✅ Inside get_phones function")
        phone_numbers = []
        phone_numbers1 = []
        phone_numbers2 = []
        for match in phonenumbers.PhoneNumberMatcher(text, "ZZ"):
        
            phone_number = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
        
            phone_numbers.append(phone_number)
        
    

        if len(phone_numbers) == 0:
            matches = re.findall(r'\b[5-9|0]\d{9,11}\b', text)
            for i in matches:
                phone_numbers1.append(i)   



        if len(phone_numbers) == 0 and len(phone_numbers1) == 0:
            regex = re.findall(r'(?:[-+() ]*\d){10,13}', text)

            for i in regex:
                phone_numbers2.append(i)

        phonenumbers_finderSet = set(phone_numbers)
        phonenumbers_finderSet1 = set(phone_numbers1)
        phonenumbers_finderSet2 = set(phone_numbers2)

    
        phone_all = phonenumbers_finderSet.union(phonenumbers_finderSet1)
        phone_all1 = phone_all.union(phonenumbers_finderSet2)
    
        return phonenumbers_finderSet,phonenumbers_finderSet1,phonenumbers_finderSet2,list(phone_all1)
    except Exception as e:
        phone = set()
        phone1 = set()
        phone2 = set()
        phone_all1=set()
        print("get_phones ",e)
        return list(phone),list(phone1),list(phone2),list(phone_all1)
