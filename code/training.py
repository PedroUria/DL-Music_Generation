import os

path = os.getcwd()[:-4] + "data/classical"

files_by_author_and_subgenre = {}
for dire in [x[0] for x in os.walk(path)][1:]:  # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    if ".mid" in " ".join(os.listdir(dire)):
        files_by_author_and_subgenre[key][dire[dire.find(key) + len(key) + 1:]] = [dire + "/" + i for i in os.listdir(dire)]
    else:
        key = dire[dire.find("classical/")+10:]
        files_by_author_and_subgenre[key] = {}
        
files_by_author = {}
for author, files in files_by_author_and_subgenre.items():
    files_by_author[author] = []
    for subgenre_files in files.values():   
        files_by_author[author] += subgenre_files
        
files_by_subgenre = {}
for files in files_by_author_and_subgenre.values():
    for key, filess in files.items():
        if key in files_by_subgenre:
            files_by_subgenre[key] += filess
        else:
            files_by_subgenre[key] = filess

# With this dicts, we should be able to get use the encoder and get the data ready for training
# TODO