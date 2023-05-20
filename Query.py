import nltk
#nltk.download()
from nltk.corpus import brown

data = brown.sents(categories='adventure')

lists =[]
for i in range(len(data)):
        str =' '.join(data[i])
        lists.append(str)

#tokenization - lists is already tokenized
from nltk.tokenize import sent_tokenize,word_tokenize
#sentences_list = [sent_tokenize(lists[i]) for i in range(len(lists))]
#print(lists)

words_list=[]
for i in lists:
       words = word_tokenize(i)
       words_list.append(words)

#stopwards
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

print(type(sw))
useful_words=[]
for i in words_list:
       for text in i:
              a = ''.join(text)
              if text not in sw:
                     useful_words.append(a)
#regex tokenizer

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('[A-Za-z]+')

print(useful_words)
useful_words = ' '.join(useful_words)
useful_text= [tokenizer.tokenize(text)]
print(useful_words)


