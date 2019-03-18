import os

renameDir = r"A:\COLLEGE\COLLEGE 2019 14th Spring Penn State\CS 488 Capstone\HTML_Forge_Tensorflow\TensorFlowTest\App\dataset\Title"

# Renames the files in a specific directory in numerical order
def renameFiles(directory):
    x = 0 #Counter for the files while renaming them
    for files in os.listdir(directory): #iterate through the files in the directory
        # if files.endswith('.png'):
        print(files) #print out the files being renamed
        # os.rename - renames the files from source to new files
        # os.path.join joins the path of the file to the new name
        os.rename(os.path.join(renameDir, files), os.path.join(renameDir, str(x) + '.png'))
        #increases the counter
        x = x+1

renameFiles(renameDir)