import re
import csv 
import os 
import argparse
import glob

class whatsapp_cleaner():

    def extract_names(self,texts):
        names = []
        for line in texts:
            name = line.split(':')[0].strip()
            if name not in names: names.append(name) 
            if len(names) == 2: break

        return names

    def clean_texts(self,filepath):

        texts = open(filepath,encoding='utf-8').read()
        pattern = r'\d+/\d+/\d+\,\s\d+:\d+\s.M\s-\s'
        lines = list(filter(None,re.split(pattern,texts)))
        names = self.extract_names(lines)

        all_texts = []
        for line in lines:
            if re.search(pattern,line) is None:
                all_texts.append(line)


        labels = []
        data = []
        messages = []
        prv_user = all_texts[0].split(':')[0].strip()

        for line in all_texts:
            for name in names:
                if line.startswith(name):
                    current_user = name
            
            txt = line.split(current_user+': ')[1].strip('\n')
            
            if current_user == prv_user:
                messages.append(txt)
            
            elif current_user != prv_user:
                data.append(' '.join(messages))
                labels.append(prv_user)
                prv_user = current_user
                messages = []
                messages.append(txt)

        data = {'Text':data,'Label':labels}
        return data,names

    def save_to_file(self,datafile,savepath):
        data,names = self.clean_texts(datafile)
        with open(savepath,'w',encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
        
        print("Saved to {}".format(savepath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for extracting data from whatsapp messages')
    parser.add_argument('datafolder',type=str,help='folder to look for whatsapp messages')
    parser.add_argument('savetofile',type=str,help='file name(s) for saving resutls',nargs='+')
    args = parser.parse_args()

    folder = os.path.join(os.getcwd(),args.datafolder)
    files = glob.glob(folder+r'\*.txt')

    if len(files) < 1:
        raise Exception('No text files found in the directory: {}'.format(folder)) 

    if len(files) > 1:
        for i,f in enumerate(files):
            print('Processing file: {}'.format(f))
            cleaner =  whatsapp_cleaner()
            cleaner.save_to_file(f,args.savetofile[i])
    else:
        cleaner =  whatsapp_cleaner()
        cleaner.save_to_file(files[0],args.savetofile)



    