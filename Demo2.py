
############################################# OPENAI AND LANGCHAIN LIBRARIES ######################################
import langchain
import os
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
import fullcontact
from fullcontact import FullContactClient
import json
############################################### END ###############################################################

############################# API AUTHENTICATION FOR LANGCHAIN AND FULL CONTACT ###################################

fullcontact_client = FullContactClient("") #Enter your Full Contact API key here!!

openai.api_key = os.environ["OPENAI_API_KEY"]

########################################### END ####################################################################

################################################ FULL CONTACT ######################################################

# Individual Information
def get_person(profiles):
    person=fullcontact_client.person.enrich(profiles=profiles)
    return person.get_summary()

# getting the info of an employee through LinkedIn URL 
data=get_person([{"url":'https://www.linkedin.com/in/marquita-ross-5b6b72192','service': 'linkedin'}])


# Flatten JSON Function for both the individual and company
def flatten_json(json_obj, parent_key='', separator='_'):
    items = {}
    for key, value in json_obj.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_json(value, new_key, separator))
        else:
            items[new_key] = value
    return items


#Function Call
flattened_data = flatten_json(data)
for key, value in flattened_data.items():
    print(f"{key}: {value}")

#Function to get the company info
def get_company(domain):
    company= fullcontact_client.company.enrich(domain=domain)
    return company.get_summary()

comp=get_company("twitter.com")

flattened_data = flatten_json(comp)
for key, value in flattened_data.items():
    print(f"{key}: {value}")

###################################################### END ############################################################

############################################## JOB RESPONSIBILITES ####################################################
llm = OpenAI(temperature=0.9)
#llm_langchain = OpenAI(model_name="text-davinci-003") 

# Run the chain only specifying the input variable.
prompt = PromptTemplate(
    input_variables=["job_role", "company_size","company_industry"],
    template="What are 5 job responsibilities someone with job role {job_role} at a company of {company_size} in {company_industry}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
output1=chain.run({
    'job_role': "Software Engineer",
    'company_size': "300",
    "company_industry":"Tech"
    })

lines = output1.split('\n')
job_roles = [lines.strip() for lines in lines if lines.strip()]
print(job_roles)

######################################################## END #####################################################################

########################################################### PAIN POINTS ##########################################################

# Run the chain only specifying the input variable.
prompt2 = PromptTemplate(
    input_variables=["job_role", "company_size","company_industry"],
    template="What are 5 job pain points someone with job role {job_role} at a company of {company_size} in {company_industry}?",
)
chain = LLMChain(llm=llm, prompt=prompt2)
output2=chain.run({
    'job_role': "Software Engineer",
    'company_size': "300",
    "company_industry":"Tech"
    })

lines2 = output2.split('\n')
pain_points = [lines2.strip() for lines2 in lines2 if lines2.strip()]
print(pain_points)

######################################################## END ##########################################################################

##################################################### EMAIL GENERATION ################################################################
llm2 = OpenAI(temperature=0.9)
prompt3 = PromptTemplate(
    input_variables=["name","product","subject","message","company", "tone", "length","job_roles","pain_points"],
    template="""Generate an email to {name} about {product}.
    The employee works at {company}.
    The job roles are {job_roles}.
    The pain points about the person's job are {pain_points}.
  The subject of the email is {subject}.
  The tone of the email should be {tone}.
  The length of the email is {length} lines.
  The body of the email is:
  {message}
  """,
)

chain = LLMChain(llm=llm2, prompt=prompt3)

# Run the chain only specifying the input variable.
print(chain.run({
    'name': "Marquita Ross",
    'subject': "Product Info",
    'product': "Auto Code Tester",
    "company":"fullcontact",
    'message': "Know about the product",
    "tone":"formal",
    "length":"8",
    "job_roles":",".join(job_roles),
    "pain_points":",".join(pain_points)
    }))

############################################################# END ########################################################################