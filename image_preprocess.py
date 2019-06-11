from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
class data_helper:
    def __init__(self):
        self.homepath=r'C:\Users\jj\Desktop\Flickr 30k\flickr30k-images'
        self.caption_path=r'D:\project\image_caption\results_20130124.token'
        self.width=500
        self.height=375
        self.batchsize=512
        self.image_list=self.file_name()
        self.caption_dict=self.caption_reader()

    def file_name(self):
        all_image=[]
        for root,dirs,files in os.walk(self.homepath):
            return files

    def getbatch_image(self):
        size=len(self.image_list)
        batch_num=(size-1)//self.batchsize+1
        for i in range(batch_num):
            start=i*self.batchsize
            end=min((i+1)*self.batchsize,size)
            sub_list=self.image_list[start:end]
            image_array=[]
            for name in sub_list:
                img = Image.open(self.homepath + '/' + name)
                new_img = img.resize((self.width, self.height), Image.BILINEAR)
                img_n = np.array(new_img)
                image_array.append(img_n)
            image_array=np.array(image_array)
            yield image_array

    def caption_reader(self):
        caption_dict={}
        file=open(self.caption_path,'r',encoding='utf-8')
        for sentence in file:
            token=sentence.split('	')
            image_name=token[0].split('.')[0]+'.jpg'
            if(image_name not in caption_dict):
                caption_dict[image_name]=[token[1]]
            else:
                caption_dict[image_name].append(token[1])
        return caption_dict

    def produce_instance(self):
        train_target=[]
        count=1
        for name in self.caption_dict:
            img = Image.open(self.homepath + '/' + name)
            new_img = img.resize((self.width, self.height), Image.BILINEAR)
            img_n = np.array(new_img)
            corrspond_caption=self.caption_dict[name]
            train_target.append([img_n,corrspond_caption])
            if(count%3000==0):
                file1=open(r'./data/'+str(count/3000)+'train_target.pkl','wb')
                pickle.dump(train_target,file1)
                file1.close()
                train_target=[]
            count+=1
        file1=open(r'./data/'+str(count/3000+1)+'train_target.pkl','wb')
        pickle.dump(train_target, file1)
        file1.close()




if __name__=='__main__':
    helper=data_helper()

    helper.produce_instance()
