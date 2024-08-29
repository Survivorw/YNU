import os
def rename(path):
    for name in os.listdir(path):
        name=os.path.join(path,name)
        for type in os.listdir(name):
            type=os.path.join(name,type)
            idx=0
            for filename in os.listdir(type):
                filename=os.path.join(type,filename)
                newname=os.path.join(type,f"{idx}.jpg")
                idx+=1
                os.rename(filename,newname)
                

path='nerual_texture'
rename(path)