#!/usr/bin/env python3

__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Majumdar, Subho"]
__email__ = "md.ahsanayub@outlook.com"
__status__ = "Prototype"


import os
import pandas as pd
from openai import OpenAI


class OpenAi:
   
   def __init__(self, api_key):
      self.api_key = api_key
      self.client = OpenAI(
         api_key = self.api_key,
      )   
      
   def get_embedding(self, text, model="text-embedding-3-small"):
      text = text.replace("\n", " ")

      # 8196 tokens are equal to approx 6000 words
      word_count = len(text.split())
      if(word_count > 6000):
         print(word_count)
         return None

      try:
         return self.client.embeddings.create(input = [text], model=model).data[0].embedding
      except:
         print(word_count)
         return None
   

class OctoAi:

   def __init__(self, api_key):
      self.api_key = api_key
      self.base_url = "https://text.octoai.run/v1"
      self.client = OpenAI(
         api_key = self.api_key,
         base_url = self.base_url,
      )

   def get_embedding(self, text, model="thenlper/gte-large"):
      text = text.replace("\n", " ")
      
      try:
         return self.client.embeddings.create(input = [text], model=model).data[0].embedding
      except:
         print(len(text))
         return None


def data_processing(obj, dataset):

   # split dataset into 1k rows for embeddings
   start = 1
   row_count = 1000
   max_count = len(dataset)
   
   while start < max_count:
      finish = start + row_count
      if finish > max_count:
         finish = max_count + 1

      # Split the dataset into row_count
      df = dataset[dataset['id'] >= start]
      df = df[df['id'] < finish]
      
      prompts = df[["text"]]

      prompts["text_embedding"] = prompts["text"].astype(str).apply(obj.get_embedding)
      prompts["label"] = df["label"].values
      prompts["id"] = df["id"].values
      
      prompts = prompts.drop(columns=['text'])
      prompts = prompts.reindex(columns=['id','text_embedding','label'])
      
      if isinstance(obj,OpenAi):
         prompts.to_csv('embeddings/openai/'+str(start)+'-'+str(finish-1)+'.csv', index=False)
      if isinstance(obj,OctoAi):
         prompts.to_csv('embeddings/octoai/'+str(start)+'-'+str(finish-1)+'.csv', index=False)
      
      print(str(finish - 1) + ' done out of ' + str(max_count))
      start = finish
      

# Driver program
if __name__ == '__main__':
   dataset = pd.read_pickle("dataset/hf_dataset.pkl")
   
   octoai = OctoAi("")
   data_processing(octoai, dataset)

   openai = OpenAi("")
   data_processing(openai, dataset)

   del dataset