import json 
import PyPDF2
 
# creating a pdf file object
pdfFileObj = open('EmployeeHandbook_January_2018.pdf', 'rb')
#pdfFileObj = open(origFileName, 'rb')
     
# creating a pdf Reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
 
# creating a pdf writer object for new pdf
pdfWriter = PyPDF2.PdfFileWriter()
emp_hb=[]     
# rotating each page
for page in range(pdfReader.numPages):
 
# creating rotated page object
     pageObj = pdfReader.getPage(page) 
     emp_hb.append(pageObj.extractText())
     #print(emp_hb)
# closing the pdf file object
pdfFileObj.close()
emp_handbook="".join(emp_hb)
#print (emp_handbook)
print ("Enter your question")
question = input()
d={}
d['distractor1']=""
d['question']=question
d['distractor3']=""
d['passage']=emp_handbook
d['correct_answer']=""
d['distractor2']=""
print (d)
with open('data_emp_hb.jsonl', 'w') as outfile:
    json.dump(d, outfile)
