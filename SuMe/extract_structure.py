##
import stanza
from typing import List
from stanza_batch import batch
import json
from nltk.corpus import stopwords

## load necessary processors
# nlp = stanza.Pipeline('en', processors='lemma,tokenize,pos,depparse')

## read input file
# target_test=open('neurologic_decoding/dataset/clean/commongen.test.tgt.txt')
target_test=open('/home/mbastan/context_home/structuralDecoding/neurologic_decoding/SuMe/commongen.sume_dev_original.tgt.txt')

lines = target_test.readlines()
target_test.close()


    
# import pickle
# try:
#     with open('stanza_documentss_test.pk','rb') as f:
#         stanza_documents=pickle.load(f)
# except:
#     ## batch document processing to increase the speed
#     stanza_documents: List[stanza.Document] = []
#     for document in batch(lines, nlp, batch_size=64): # Default batch size is 32
#         stanza_documents.append(document)
#     ## save the documents on a file
#     with open('stanza_documents_test.pk','wb') as f:
#         pickle.dump(stanza_documents,f)   
neurologic_constraints_f=open('../../dgx_neurologic_decoding/dataset/clean/constraint/sume_dev_v3.constraint.json','w')
import re
## write all extracted structures on a a file
with open('../dataset/clean/constraint/sume_dev_v3.structure.tgt.json','w') as f:    
    ## write only non stop words which are nsubj, obj, root on a file
    with open('../dataset/clean/constraint/sume_dev_original_v3.constraint.json','w') as g_ori:
        ## write reverse structure
        with open('../dataset/clean/constraint/sume_dev_reverse_v3.constraint.json','w') as g_rev:
            stop_words = set(stopwords.words('english'))
            # for doc in  stanza_documents:
            for line in lines:
                words = line.split()
                # sent = doc.sentences[0]
                subj = re.search(r'<re>(.*?)<er>', line).group(1).strip().lower()
                obj = re.search(r'<el>(.*?)<le>', line).group(1).strip().lower()
                arr_all = []
                arr_ori = []
                arr_rev = []
                arr_neurologic=[]
                arr_neurologic.append([subj,'<re> %s <er>'%subj])
                arr_neurologic.append([obj,'<el> %s <le>'%obj])
                # arr_ori.append([subj,'subj'])
                # arr_rev.append([subj,'obj'])
                # if len(subj.split())>1:
                for w in subj.split():
                        if w in stop_words:
                            continue
                        arr_ori.append([w,'nsubj'])
                        arr_rev.append([w,'obj'])
                # arr_ori.append([obj,'obj'])
                # arr_rev.append([obj,'nsubj'])
                # if len(obj.split()) > 1:
                for w in obj.split():
                        if w in stop_words:
                                continue
                        arr_ori.append([w,'obj'])
                        arr_rev.append([w,'nsubj'])
                
                json.dump(arr_ori,g_ori)
                g_ori.write('\n')
                
                json.dump(arr_rev,g_rev)
                g_rev.write('\n')
                
                json.dump(arr_neurologic,neurologic_constraints_f)
                neurologic_constraints_f.write('\n')
                # for word in sent.words:
                    # print('word',word)
                    # if not word.lemma:
                        # continue
                    # arr_all.append([word.lemma,word.deprel])
                    # if word.lemma.lower() in stop_words:
                        # continue
                    # if word.deprel in sent_map:
                        # continue
                    # if word.deprel == 'nsubj' or word.deprel == 'obj' or word.deprel == 'root' or word.deprel == 'obl':
                    #     if( word.deprel == 'obl' and 'obj' in sent_map) or ( word.deprel == 'obj' and 'obl' in sent_map):
                    #         continue
                        
                        # arr_ori.append([word.lemma,word.deprel])
                        # if word.deprel == 'nsubj':
                        #     arr_rev.append([word.lemma,'obj'])
                        # elif (word.deprel == 'obj' or  word.deprel == 'obl'):
                        #     arr_rev.append([word.lemma,'nsubj'])
                        # else:
                        #     arr_rev.append([word.lemma,'root'])
                        
                ## write all structures
                # json.dump(arr_all,f)
                # f.write('\n')
                
                # json.dump(arr_ori,g_ori)
                # g_ori.write('\n')
                
                # json.dump(arr_rev,g_rev)
                # g_rev.write('\n')
neurologic_constraints_f.close()               
                