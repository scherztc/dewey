import re
import pandas as pd

df =pd.read_csv('Titanic Numbers.csv')
df

while True:
  messages = [] 
  messages.append(
  {
    'role':'user',
    'content':'''
    Here is the scheme of my data
    PassnergerId, Survived, PCclass, Name
    Return the answer in python code only.
    For your info I have loaded a CSV file.
    '''
  })

prompt = input('\nAsk a question: ')

if prompt = "quit":
   break

messages.append (
{
   'role':'user',
    'content':'''
    Here is the scheme of my data
    PassnergerId, Survived, PCclass, Name
    Return the answer in python code only.
    For your info I have loaded a CSV file.
    '''
    })

completion = client.chat.completions.create(
   model = client_name,
   messages = messages,
   max_tokens = 1024,
   temperature =0

print(f")

