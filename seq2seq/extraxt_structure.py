##
import stanza
from typing import List
from stanza_batch import batch
import json
from nltk.corpus import stopwords

## load necessary processors
nlp = stanza.Pipeline('en', processors='lemma,tokenize,pos,depparse')

## read input file
target_test=open('neurologic_decoding/dataset/clean/commongen.test.tgt.txt')
lines = target_test.readlines()
target_test.close()

## batch document processing to increase the speed
stanza_documents: List[stanza.Document] = []
for document in batch(lines, nlp, batch_size=64): # Default batch size is 32
    stanza_documents.append(document)
    
import pickle
## save the documents on a file
with open('stanza_documents_test.pk','wb') as f:
    pickle.dump(stanza_documents,f)   


## write all extracted structures on a a file
with open('neurologic_decoding/dataset/clean/constraint/test.structure.tgt.json','w') as f:    
    ## write only non stop words which are nsubj, obj, root on a file
    with open('neurologic_decoding/dataset/clean/constraint/test_original.constraints.tgt.json','w') as g_ori:
        ## write reverse structure
        with open('neurologic_decoding/dataset/clean/constraint/test_reverse.constraints.tgt.json','w') as g_rev:
            stop_words = set(stopwords.words('english'))
            for doc in  stanza_documents:
                sent = doc.sentences[0]
                arr_all = []
                arr_ori = []
                arr_rev = []
                sent_map ={}
                for word in sent.words:
                    arr_all.append([word.lemma,word.deprel])
                    if word.lemma.lower() in stop_words:
                        continue
                    if word.deprel in sent_map:
                        continue
                    if word.deprel == 'nsubj' or word.deprel == 'obj' or word.deprel == 'root' or word.deprel == 'obl':
                        if( word.deprel == 'obl' and 'obj' in sent_map) or ( word.deprel == 'obj' and 'obl' in sent_map):
                            continue
                        sent_map[word.deprel]=word.lemma
                        arr_ori.append([word.lemma,word.deprel])
                        if word.deprel == 'nsubj':
                            arr_rev.append([word.lemma,'obj'])
                        elif (word.deprel == 'obj' or  word.deprel == 'obl'):
                            arr_rev.append([word.lemma,'nsubj'])
                        else:
                            arr_rev.append([word.lemma,'root'])
                        
                ## write all structures
                json.dump(arr_all,f)
                f.write('\n')
                
                json.dump(arr_ori,g_ori)
                g_ori.write('\n')
                
                json.dump(arr_rev,g_rev)
                g_rev.write('\n')
                
                