import json
import os, glob
from datetime import datetime
from langdetect import detect, detect_langs


if __name__=='__main__':
    path = '../data/elezioni_politiche/raw'
    for filename in glob.glob(os.path.join(path, '*.json')):
        print(filename)
        trash=0
        with open(filename, 'r') as f: # open in readonly mode
            d=json.load(f)
            dump=[]
            for element in d:
                language1=""
                language2=""

                try:
                    #if (len(element['reviewTitle']) > 0 ):
                    #   language1 = detect(element['reviewTitle'])
                    if len(element['text']) > 0:
                       language2 = detect(element['text'])

                    if language2 == 'en':
                           dump.append(element)

                    else:
                        trash+=1

                except:
                    trash+=1



        print("Trashed:",trash)
        with open(filename, 'w', encoding='utf-8') as f1:
            json.dump(dump, f1, ensure_ascii=False, indent=4)

    print("DONE")