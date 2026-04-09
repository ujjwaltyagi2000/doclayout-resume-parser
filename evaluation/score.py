def match_strings(string1, string2):
    string1_lower = string1.lower()
    string2_lower = string2.lower()


    words1 = string1_lower.split()
    words2 = string2_lower.split()

    for word1 in words1:
        for word2 in words2:
            if word1 in word2 or word2 in word1:
                return True
    return False

def calculate_resume_score(standard_font_style_flag,multiple_font_style,multiple_font_size,content_size_flag,font_colors_Total,standard_color_flag,actionwords_total,actionwords_total_negative,total_repeated_actionwords_negative,Bullets_Total,standard_bullet_flag,total_repeated_actionwords,ats_date,dates_nonAts,clean_measurable,total_word_count,email_finderSet,phonenumbers_finderSet,images,linkedIn_flag,pages_count,personalPronouns,tables_flag,Work_Project_Headings,EduSkill_Headings,NRlength,ORlength_db,Skills_Total,standard_match_headings_count,sectionMapCount,phone_all1,actualHeadingsCount,experience,fillerwords_total,output_voice,repeated_words,file_name, name_output, ORlength ):
    try:
    
    
        score_array = []
        scoring  = 0
        scoring1 = 0
        if standard_font_style_flag == 1:
            scoring += 3
            scoring1 = 3
        else:
            scoring += 0
            scoring1 = 0

        score_array.append(scoring1)  
        print(scoring)


        scoring1 = 0
        if multiple_font_style == 1 or multiple_font_style == 2:
            scoring += 2
            scoring1 = 2
        elif multiple_font_style == 3 or multiple_font_style == 4:    
            scoring += 1
            scoring1 = 1

        else:
            scoring += 0
            scoring1 = 0   
        

        score_array.append(scoring1)
        print(scoring)


        scoring1 = 0
        if multiple_font_size == 1 or multiple_font_size>6:
            scoring -= 2
            scoring1 = -2
        if multiple_font_size == 3  or multiple_font_size == 4:    
            scoring += 5
            scoring1 = 5
        if multiple_font_size == 5  or multiple_font_size == 6 or multiple_font_size == 2:    
            scoring += 2
            scoring1 = 2

        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if content_size_flag == 1:
            scoring += 6
            scoring1 = 6
        else:
            scoring -= 3 
            scoring1 = -3
        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if font_colors_Total < 5:
            scoring += 1
            scoring1 = 1
        elif font_colors_Total == 5:
            scoring += 1
            scoring1 = 1
        else:
            scoring += 0    
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if standard_color_flag == 1:
            scoring += 1
            scoring1 = 1
        else:
            scoring += 0    
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if experience == 0:
            if len(Work_Project_Headings) == 2:
                scoring += 4
                scoring1 = 4
            elif len(Work_Project_Headings) == 1:
                scoring += 4
                scoring1 = 4
            else:
                scoring -= 5
                scoring1 = -5

        elif experience <2:
            if len(Work_Project_Headings) == 2:
                scoring += 4
                scoring1 = 4
            elif len(Work_Project_Headings) == 1:
                scoring += 2
                scoring1 = 2
            else:
                scoring -= 5
                scoring1 = -5

        elif experience >= 2 and experience<=5:
            if len(Work_Project_Headings) == 2:
                scoring += 4
                scoring1 = 4
            elif len(Work_Project_Headings) == 1:
                scoring += 2
                scoring1 = 2
            else:
                scoring -= 5
                scoring1 = -5

        elif experience >5:
            if len(Work_Project_Headings) == 2:
                scoring += 4
                scoring1 = 4
            elif len(Work_Project_Headings) == 1:
                scoring += 2
                scoring1 = 2
            else:
                scoring -= 5
                scoring1 = -5
        else:
            scoring+=0
            scoring1 = 0


        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if len(EduSkill_Headings) == 2:
            scoring += 5
            scoring1 = 5
        elif len(EduSkill_Headings) == 1:
            scoring += 2
            scoring1 = 2
        else:
            scoring -= 5
            scoring1 = -5

        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if experience==0:
            if actionwords_total > 10:
                scoring += 9
                scoring1 = 9

            elif actionwords_total >= 4 and actionwords_total <= 10: 
                value = ((actionwords_total)/11)*9
                scoring += value
                scoring1 = value
            else:
                scoring -= 2
                scoring1 = -2        
        elif experience<2:
            if actionwords_total > 10:
                scoring += 9
                scoring1 = 9
            elif actionwords_total >= 4 and actionwords_total <= 10: 
                value = ((actionwords_total)/11)*9
                scoring += value
                scoring1 = value
            else:
                scoring -= 2
                scoring1 = -2
        elif experience>=2 and experience<=5:
            if actionwords_total > 10:
                scoring += 9
                scoring1 = 9

            elif actionwords_total >= 4 and actionwords_total <= 10: 
                value = ((actionwords_total)/11)*9
                scoring += value
                scoring1 = value

            else:
                scoring -= 2
                scoring1 = -2
        elif experience>5:
            if actionwords_total > 10:
                scoring += 9
                scoring1 = 9

            elif actionwords_total >= 4 and actionwords_total <= 10: 
                value = ((actionwords_total)/11)*9
                scoring += value
                scoring1 = value
            else:
                scoring -= 2
                scoring1 = -2
        else:
            scoring+=0
            scoring1 = 0
        
        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if NRlength > 2:
            scoring -= 6
            scoring1 = -6
        elif NRlength == 2:      
            scoring -= 4
            scoring1 = -4

        elif NRlength == 1:    
            scoring -= 2
            scoring1 = -2
        else:
            scoring += 0    
            scoring1 = 0
        print(scoring)
        
        score_array.append(scoring1)

#changed
        scoring1 = 0
        if experience==0:
            if ORlength > 2:
                scoring += 5
                scoring1 = 5
            elif ORlength == 1 or ORlength == 2:
                scoring += 2
                scoring1 = 2
            else:
                scoring += 0
                scoring1 = 0

        elif experience<2:
            if ORlength > 2:
                scoring += 5
                scoring1 = 5
            elif ORlength == 1 or ORlength == 2:
                scoring += 3
                scoring1 = 3
            else:
                scoring += 0
                scoring1 = 0

        elif experience>=2 and experience<=5:
            if ORlength > 1:
                scoring += 5
                scoring1 = 5
            elif ORlength == 1 :
                scoring += 2
                scoring1 = 2
            else:
                scoring += 0
                scoring1 = 0
        
        elif experience>5:
            if ORlength > 1:
                scoring += 5
                scoring1 = 5
            elif ORlength == 1 :
                scoring += 2
                scoring1 = 2
            else:
                scoring += 0
                scoring1 = 0
        else:
            scoring+=0
            scoring1 = 0


        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if experience == 0:
            if total_repeated_actionwords > 2:
                scoring -= 3
                scoring1 = -3
            elif total_repeated_actionwords == 2:            
                scoring -= 2
                scoring1 = -2
            elif total_repeated_actionwords == 1:            
                scoring -= 2
                scoring1 = -2
            else:
                scoring += 0
                scoring1 = 0

        elif experience <2:
            if total_repeated_actionwords > 2:
                scoring -= 3
                scoring1 = -3
            elif total_repeated_actionwords == 2:            
                scoring -= 2
                scoring1 = -2
            elif total_repeated_actionwords == 1:            
                scoring -= 2
                scoring1 = -2
            else:
                scoring += 0
                scoring1 = 0
        
        elif experience >=2 and experience<=5:
            if total_repeated_actionwords > 2:
                scoring -= 3
                scoring1 = -3
            elif total_repeated_actionwords == 2:            
                scoring -= 2
                scoring1 = -2
            elif total_repeated_actionwords == 1:            
                scoring -= 2
                scoring1 = -2
            else:
                scoring += 0
                scoring1 = 0
        elif experience >5:
            if total_repeated_actionwords > 2:
                scoring -= 3
                scoring1 = -3
            elif total_repeated_actionwords == 2:            
                scoring -= 2
                scoring1 = -2
            elif total_repeated_actionwords == 1:            
                scoring -= 2
                scoring1 = -2
            else:
                scoring += 0
                scoring1 = 0
        else :
            scoring+=0
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if Bullets_Total == 0 or Bullets_Total > 2:
            scoring -= 4
            scoring1 = -4
        elif Bullets_Total == 2:
            scoring -= 3
            scoring1 = -3
        else:
            scoring += 0 
            scoring1 = 0   


        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if standard_bullet_flag == 1:
            scoring += 3
            scoring1 = 3
        else:
            scoring += 0
            scoring1 = 0

        print(scoring)     
        score_array.append(scoring1)

        ats_length = 1 if len(ats_date) > 0 else 0
        nonats_length = 1 if len(dates_nonAts) > 0 else 0

        scoring1 = 0
        if ats_length == 1:
            scoring += 5
            scoring1 = 5
        else:
            scoring += 0 
            scoring1 = 0     

        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0
        if nonats_length == 1:
            scoring -= 3
            scoring1 = -3
        else:
            scoring += 0       
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if actionwords_total_negative > 2:
            scoring -= 4
            scoring1 = -4
        elif actionwords_total_negative == 1 or actionwords_total_negative == 2:
            scoring -= 2
            scoring1 = -2
        else:
            scoring += 0
            scoring1 = 0

        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if total_repeated_actionwords_negative > 2:
            scoring -= 2
            scoring1 = -2
        elif total_repeated_actionwords_negative == 1 or total_repeated_actionwords_negative == 2:
            scoring -= 1
            scoring1 = -1

        else:
            scoring += 0
            scoring1 = 0   

        print(scoring)   
        score_array.append(scoring1)

        scoring1 = 0
        if linkedIn_flag == 1:
            scoring += 4
            scoring1 = 4
        else:
            scoring -= 4    
            scoring1 = -4


        print(scoring)
        score_array.append(scoring1)
#changed
        scoring1 = 0
        if experience==0:

            if len(clean_measurable) >=12:
                scoring += 9
                scoring1 = 9

            elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                new_score = ((len(clean_measurable))/12)*9   
                scoring += new_score
                scoring1 = new_score

            elif len(clean_measurable)<=4:
                scoring -= 3 
                scoring1 = -3   
            else:
                scoring+=0
                scoring1 = 0
        elif experience<2:

            if len(clean_measurable) >=12:
                scoring += 9
                scoring1 = 9

            elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                new_score = ((len(clean_measurable))/12)*9   
                scoring += new_score
                scoring1 = new_score
            elif len(clean_measurable)<=4:
                scoring -= 3    
                scoring1 = -3
            else:
                scoring+=0
                scoring1 = 0
        elif experience>=2 and experience<=5:

            if len(clean_measurable) >=12:
                scoring += 9
                scoring1 = 9

            elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                new_score = ((len(clean_measurable))/12)*9   
                scoring += new_score
                scoring1 = new_score

            elif len(clean_measurable)<=4:
                scoring -= 3    
                scoring1 = -3
            else:
                scoring+=0
                scoring1 = 0
        elif experience>5:

            if len(clean_measurable) >=12:
                scoring += 9
                scoring1 = 9

            elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                new_score = ((len(clean_measurable))/12)*9   
                scoring += new_score
                scoring1 = new_score
            elif len(clean_measurable)<=4:
                scoring -= 3    
                scoring1 = -3
            else:
                scoring+=0
                scoring1 = 0
        else:
            scoring+=0
            scoring1 = 0




        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if experience==0:

            if total_word_count <= 150:
                scoring -= 8
                scoring1 = -8
            elif total_word_count > 150 and total_word_count <= 165: 
                scoring -= 6
                scoring1 = -6
            elif total_word_count > 165 and total_word_count <= 200: 
                scoring += 2
                scoring1 = 2
            elif total_word_count > 200 and total_word_count <= 300: 
                scoring += 4
                scoring1 = 4
            elif total_word_count > 300 and total_word_count <= 450: 
                scoring += 8
                scoring1 = 8
            elif total_word_count > 450 and total_word_count <= 600: 
                scoring += 4
                scoring1 = 4
            else:
                scoring -= 8
                scoring1 = -8

        elif experience<2:

            
            if total_word_count <= 165: 
                scoring -= 8
                scoring1 = -8
            elif total_word_count > 165 and total_word_count <= 250: 
                scoring -= 4
                scoring1 = -4
            elif total_word_count > 250 and total_word_count <= 350: 
                scoring += 6
                scoring1 = 6
            elif total_word_count > 350 and total_word_count <= 500: 
                scoring += 8
                scoring1 = 8
            elif total_word_count > 500 and total_word_count <= 700: 
                scoring += 4
                scoring1 = 4
            else:
                scoring -= 8
                scoring1 = -8

        elif experience>=2 and experience<=5:

            if total_word_count <= 165: 
                scoring -= 8
                scoring1 = -8
            elif total_word_count > 165 and total_word_count <= 250: 
                scoring -= 4
                scoring1 = -4
            elif total_word_count > 250 and total_word_count <= 400: 
                scoring += 4
                scoring1 = 4
            elif total_word_count > 400 and total_word_count <= 750: 
                scoring += 8
                scoring1 = 8
            elif total_word_count > 750 and total_word_count <= 900: 
                scoring += 4
                scoring1 = 4
            else:
                scoring -= 8
                scoring1 = -8


        elif experience>5:

            if total_word_count <= 165:
                scoring -= 8
                scoring1 = -8
            elif total_word_count > 165 and total_word_count <= 200: 
                scoring -= 4
                scoring1 = -4
            elif total_word_count > 200 and total_word_count <= 300: 
                scoring -= 2
                scoring1 = -2
            elif total_word_count > 300 and total_word_count <= 450: 
                scoring += 2
                scoring1 = 2
            elif total_word_count > 450 and total_word_count <= 950: 
                scoring += 8
                scoring1 = 8
            elif total_word_count > 950 and total_word_count <= 1000: 
                scoring += 6
                scoring1 = 6
            
            else:
                scoring -= 8
                scoring1 = -8
        
        else :
            scoring+=0
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)

#changed
        scoring1 = 0
        if sectionMapCount == 0: 
            scoring += 0
            scoring1 = 0
        elif sectionMapCount == 1: 
            scoring -= 1
            scoring1 = -1 
        elif sectionMapCount == 2: 
            scoring -= 3
            scoring1 = -3
        else:
            scoring -= 4 
            scoring1 = -4   
        
        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if standard_match_headings_count<=3:
            scoring+=0
            scoring1 = 0
        else :
            new_score = (standard_match_headings_count/actualHeadingsCount) * 5
            scoring += new_score    
            scoring1 = new_score       
        print(scoring)
        score_array.append(scoring1)


        email_flag = 1 if len(email_finderSet) > 0 else 0
        phone_flag = 1 if len(phone_all1) > 0 else 0

        scoring1 = 0
        if email_flag == 1:
            scoring += 1
            scoring1 = 1
        else:
            scoring -= 5
            scoring1 = -5    
    

        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if phone_flag == 1:
            scoring += 1
            scoring1 = 1
        else:
            scoring -= 5
            scoring1 = -5 
    
        print(scoring)
        score_array.append(scoring1)
#changed
        scoring1 = 0
        if experience==0:

            if Skills_Total >= 5 and Skills_Total < 18:
                new_score =  ((Skills_Total)/(18))*9    
                scoring += new_score 
                scoring1 = new_score

            elif Skills_Total >= 18 and Skills_Total <= 55:
                scoring += 9
                scoring1 = 9
            elif Skills_Total > 55:
                scoring += 4.5
                scoring1 = 4.5
            elif Skills_Total<=4:
                scoring -= 6  
                scoring1 = -6  
            else:
                scoring+=0
                scoring1 = 0

        elif experience<2:

            if Skills_Total >= 5 and Skills_Total < 20:
                new_score =  ((Skills_Total)/(20))*9    
                scoring += new_score 
                scoring1 = new_score

            elif Skills_Total >= 20 and Skills_Total <= 55:
                scoring += 9
                scoring1 = 9

            elif Skills_Total > 55:
                scoring += 4.5
                scoring1 = 4.5
            elif Skills_Total<=4:
                scoring -= 6  
                scoring1 = -6  
            else:
                scoring+=0
                scoring1 = 0

        elif experience>=2 and experience<=5:

            if Skills_Total >= 5 and Skills_Total < 24:
                new_score =  ((Skills_Total)/(24))*9    
                scoring += new_score 
                scoring1 = new_score

            elif Skills_Total >= 24 and Skills_Total <= 65:
                scoring += 9
                scoring1 = 9

            elif Skills_Total > 65:
                scoring += 4.5
                scoring1 = 4.5
            elif Skills_Total<=4:
                scoring -= 6  
                scoring1 = -6  
            else:
                scoring+=0
                scoring1 = 0
        elif experience>5:

            if Skills_Total >= 5 and Skills_Total < 24:
                new_score =  ((Skills_Total)/(24))*9    
                scoring += new_score 
                scoring1 = new_score

            elif Skills_Total >= 24 and Skills_Total <= 65:
                scoring += 9
                scoring1 = 9

            elif Skills_Total > 65:
                scoring += 4.5
                scoring1 = 4.5

            elif Skills_Total<=4:
                scoring -= 6  
                scoring1 = -6  
            else:
                scoring+=0
                scoring1 = 0
        else:
            scoring+=0
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)
        
        scoring1 = 0 
        if images =="No images found in the PDF file.":
            scoring +=0
            scoring1 = 0
        else:
            scoring -=4
            scoring1 = -4
        
        print(scoring)
        score_array.append(scoring1)



        scoring1 = 0 
        if len(repeated_words) > 3:
            scoring -= 3
            scoring1 = -3
        elif len(repeated_words) >= 1 and len(repeated_words) <= 3:
            scoring -= 1
            scoring1 = -1
        
        else:
            scoring += 3
            scoring1 = 3
        print(scoring)
        score_array.append(scoring1)
    
        


        scoring1 = 0 
        if match_strings(file_name, name_output):
            scoring += 1
            scoring1 = 1
        else:
            scoring += 0
            scoring1 = 0
        print(scoring)
        print("Name")
        score_array.append(scoring1)





        scoring1 = 0 
        if len(output_voice) > 2:
            scoring -= 3
            scoring1 = -3
        elif len(output_voice) == 1 or len(output_voice) == 2:
            scoring -= 1
            scoring1 = -1
        
        else:
            scoring += 3
            scoring1 = 3
        print(scoring)
        score_array.append(scoring1)


        scoring1 = 0 
        if fillerwords_total > 3:
            scoring -= 3
            scoring1 = -3
        elif fillerwords_total >= 1 and fillerwords_total <= 3:
            scoring -= 1
            scoring1 = -1
        
        else:
            scoring += 2
            scoring1 = 2
        print(scoring)
        score_array.append(scoring1)




    #changed
        scoring1 = 0
        if experience == 0:
            if pages_count == 1 :
                scoring += 5
                scoring1 = 5
            elif pages_count == 3 or pages_count == 2:
                scoring -= 4
                scoring1 = -4
            else:
                scoring -= 6 
                scoring1 = -6  
        
        elif experience<2:
            if pages_count == 1 or pages_count == 2:
                scoring += 5
                scoring1 = 5
            elif pages_count == 3:
                scoring -= 4
                scoring1 = -4
            else:
                scoring -= 6 
                scoring1 = -6  
        elif experience>=2 and experience<=5:
            if pages_count == 1 or pages_count == 2:
                scoring += 5
                scoring1 = 5
            elif pages_count == 3:
                scoring -= 4
                scoring1 = -4
            else:
                scoring -= 6 
                scoring1 = -6  
        elif experience>5:
            if pages_count == 1 or pages_count == 2:
                scoring += 5
                scoring1 = 5
            elif pages_count == 3:
                scoring -= 4
                scoring1 = -4
            else:
                scoring -= 6 
                scoring1 = -6  
        else:
            scoring+=0
            scoring1 = 0
        print(scoring)
        score_array.append(scoring1)

        personalPronouns_flag = 1 if len(personalPronouns)>0 else 0
        scoring1 = 0
        if personalPronouns_flag==1:
            scoring -= 3
            scoring1 = -3
        else:
            scoring += 0
            scoring1 = 0
        
        print(scoring)
        score_array.append(scoring1)

        scoring1 = 0
        if tables_flag==1:
            scoring -= 4
            scoring1 = -4
        else:
            scoring -= 0
            scoring1 = 0
    
        score_array.append(scoring1)
        print(scoring)
        print("net")
    
# final scoring changed
        scoring1 = 0
        if scoring >=0 and scoring<10 :
            if total_word_count>=0 and total_word_count<300:
                scoring+=4
                scoring1 = 4
            else:
                scoring+=7 
                scoring1 = 7

    

        if scoring >=10 and scoring<15 :
            if total_word_count>=0 and total_word_count<300:
                scoring+=2
                scoring1 = 2
            else:
                scoring+=2  
                scoring1 = 2    
        


        if scoring >=20 and scoring<25 and total_word_count>500:
            scoring+=3
            scoring1 = 3

    

        if scoring >=25 and scoring<30 :
            if total_word_count>=0 and total_word_count<300:
                scoring-=4
                scoring1 = -4
            elif total_word_count>=300 and total_word_count<500:
                scoring-=2
                scoring1 = -2
            else:
                scoring+=0
                scoring1 = 0

    

        if scoring >=30 and scoring<35 :
            if total_word_count>=0 and total_word_count<250:
                scoring-=3
                scoring1 = -3
            elif total_word_count>=250 and total_word_count<400:
                scoring+=0
                scoring1 = 0
            else:
                scoring+=3
                scoring1 = 3
            
        

        if scoring >=35 and scoring<40 :
            if total_word_count>=0 and total_word_count<275:
                scoring-=8
                scoring1 = -8
            elif total_word_count>= 275 and total_word_count<400:
                scoring-=3
                scoring1 = -3
            elif total_word_count>=400 and total_word_count<500:
                scoring-=1.5
                scoring1 = -1.5
            else:
                scoring+=0  
                scoring1 = 0


    

        if scoring >=45 and scoring<50 : 
            if total_word_count>=0 and total_word_count<300:
                scoring-=11
                scoring1 = -11
            elif total_word_count>=300 and total_word_count<400:
                scoring-=6
                scoring1 = -6
            elif total_word_count>=400 and total_word_count<500:
                scoring-=4
                scoring1 = -4
            else:
                scoring+=0  
                scoring1 = 0

        if scoring >=50 and scoring<60: 
            if total_word_count>=0 and total_word_count<300:
                scoring-=5
                scoring1 = -5
            elif total_word_count>=300 and total_word_count<400:
                scoring-=3
                scoring1 = -3
            else:
                scoring+=0   
                scoring1 = 0  
        score_array.append(scoring1)
        print(scoring)   
        x = round(scoring)
        return x,score_array



    except Exception as e:
        print("Error generating Resume Score: ",e)
        return 180,[]