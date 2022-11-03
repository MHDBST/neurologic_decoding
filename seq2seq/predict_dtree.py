def get_dep_tree_connections(mapp,entity):
    connection_arr=[]
    if entity in mapp:
        for arr in mapp[entity]:
            current_word = entity
            deprel = arr[0]
            head = arr[1]
            connection_arr.append([current_word,deprel,head])
            while head != 'root' and head.lower() != current_word:
                current_word = head.lower()
                if current_word not in mapp:
                    break
                for arr1 in mapp[current_word]:
                    deprel = arr1[0]
                    head = arr1[1]
                    connection_arr.append([current_word,deprel,head])
                    if [current_word,deprel,head] in connection_arr:
                        break
                if connection_arr.count([current_word,deprel,head]) > 2:
                    break
        return connection_arr
    return None

def get_tree(nlp,sentence):
    # print('input here is>>',sentence)
    sentence=sentence.strip()
    if not sentence:
        return []
    doc = nlp(str(sentence))
    # for sent in doc:
    sent=doc.sentences[0]
    a_ents=[]
    a_map={}
    for word in sent.words:
        if word.lemma not in a_map:
            a_map[word.lemma]=[[word.deprel,sent.words[word.head-1].text if word.head>0 else "root"]]
        else:
            a_map[word.lemma].append([word.deprel,sent.words[word.head-1].text if word.head>0 else "root"])
        if word.deprel=='nsubj' or word.deprel=='obj':
            a_ents.append(word.lemma)
        
    a_connection_arrs=[]
    for ent in a_ents:
        a_connection_arr=get_dep_tree_connections(a_map,ent)
        a_connection_arrs.append(a_connection_arr)
    verb_map={}
    # print('a_connection_arrs',a_connection_arrs)
    for arrr in a_connection_arrs:
        for arr in arrr:
            # ["person", "nsubj", "riding"], ["bicycle", "obj", "riding"]
            if arr[2]!='root':
                if arr[2] in verb_map:
                    verb_map[arr[2]][arr[1]]=arr[0]
                else:
                    verb_map[arr[2]]={arr[1]:arr[0]}
    a_connection_arrs=[]
    for item in verb_map:
        try:
            a_connection_arrs.append([verb_map[item]['nsubj'],item,verb_map[item]['obj']])
        except:
            pass
            
                
        # a_connection_arrs.append(a_connection_arr)
    # print('a_connection_arr:',a_connection_arrs)
    # print('verb_map',verb_map)
    return(a_connection_arrs)

def get_tree_1(nlp,sentence):
    doc = nlp(sentence)
    # print('doc',doc)
    # for sent in doc:
    sent=doc.sentences[0]
    a_ents=[]
    a_map={}
    for word in sent.words:
        if word.text.lower() not in a_map:
            a_map[word.text.lower()]=[[word.deprel,sent.words[word.head-1].text if word.head>0 else "root"]]
        else:
            a_map[word.text.lower()].append([word.deprel,sent.words[word.head-1].text if word.head>0 else "root"])
        if word.deprel=='nsubj' or word.deprel=='obj':
            a_ents.append(word.text.lower())
        
    a_connection_arrs=[]
    for ent in a_ents:
        a_connection_arr=get_dep_tree_connections(a_map,ent)
        a_connection_arrs.append(a_connection_arr)
    # print('a_connection_arr:',a_connection_arrs)
    return(a_connection_arrs)
