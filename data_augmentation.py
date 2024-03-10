from transformers import MarianMTModel, MarianTokenizer
import random
import nltk

# Sentence to sentence: str -> str

def back_translate_init():
  en2fr_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
  en2fr_tkn = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
  fr2en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
  fr2en_tkn = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
  return en2fr_model, en2fr_tkn, fr2en_model, fr2en_tkn

def back_translate(sentence, en2fr_model, en2fr_tkn, fr2en_model, fr2en_tkn):
  formated = [">>{}<< {}".format("fr", sentence)]
  translated = en2fr_model.generate(**en2fr_tkn(formated, return_tensors="pt", padding=True))
  temp = [en2fr_tkn.decode(t, skip_special_tokens=True) for t in translated][0]

  formated = [">>{}<< {}".format("en", temp)]
  translated = fr2en_model.generate(**fr2en_tkn(formated, return_tensors="pt", padding=True))
  similar_sen = [fr2en_tkn.decode(t, skip_special_tokens=True) for t in translated][0]

  if similar_sen == sentence: return None # Returns None if input sentence is unchanged
  return similar_sen

#EXAMPLE
#en2fr_model, en2fr_tkn, fr2en_model, fr2en_tkn = back_translate_init()
#sentence = "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free"
#print(back_translate(sentence, en2fr_model, en2fr_tkn, fr2en_model, fr2en_tkn))

def synonym_replace_init():
  random.seed(1)
  nltk.download('wordnet')

def synonym_replace(sentence):
  substrings = sentence.split()
  indices = random.sample(range(len(substrings)), random.randint(1, max(1, len(substrings) // 2)))
  for i in indices:
    syn = wordnet.synsets(substrings[i])
    if len(syn) == 0 : continue
    lem = syn[random.randint(0, (len(syn) - 1) // 2)].lemmas()
    substrings[i] = lem[random.randint(0, (len(lem) - 1) // 2)].name().replace("_", " ")

  similar_sen = " ".join(substrings)
  if similar_sen == sentence: return None # Returns None if input sentence is unchanged (retry may succeed due to randomness)
  return similar_sen

#EXAMPLE
#synonym_replace_init()
#sentence = "The guy did some bitching but I acted like i'd be interested in buying something else next week and he gave it to us for free"
#print(synonym_replace(sentence))