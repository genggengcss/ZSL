from nltk.corpus import wordnet as wn

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))
def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s



# get hypernyms and hyponyms

wnid = "n00479887"
wnid_syn = getnode(wnid)  # get wn synset
# for wn_per in wnid_syn.hypernyms():
#     print(wn_per)
#     wnid_1_hop = getwnid(wn_per)
#     print(wnid_1_hop)
# for wn_per in wnid_syn.hyponyms():
#     wnid_1_hop = getwnid(wn_per)


# get corresponding wname
wn_name = wnid_syn.lemma_names()[0]
print(wn_name)


